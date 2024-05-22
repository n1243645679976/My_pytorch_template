# README
## Introduction
This repository contains features like easy model modification, joint training, subset training, pair selecting during training and model loading.
## Commands for training
For example:
```
bash run_utmos_baseline.sh --stages "1 3" --test_iter 60000 --extract_feature_online True --outdir output/some_thing_exp001 --train_conf conf/UTMOS/utmos_v1_pref4_noaug.yaml --test_conf conf/UTMOS/utmos_v1_pref4_noaug_test.yaml --train data_1,data_2,data_3*28 --dev data_dev --test data_test
```
arguments includes:
```
stages: in the run.sh, we have 4 stages, stage 0 and 2 are for feature extraction on datasets, stage 1 and 3 for model training and test. we only
test_iter: use iteration = 60000 for testing
extract_feature_online: if True, then we skip feature extraction on stage 0 and 2. The model will extract features online.
outdir: the experiment result will be under this directory, including used data, loss, saved model, training argument and testing result.
train_conf, test_conf: configuration for training and testing
train: training dataset, including joint training like data_1,data_1_another,data_2*28 meaning using three datasets data_1, data_2 and data_3 under directories data/data_1, data/data_2 and data/data_3, respectively. The selection probability is 1:1:28
test: test dataset
```
## Configure
This configuration file will be used in the argument in train_conf, test_conf.
For example `conf/UTMOS/utmos_v1_pref4_noaug.yaml`:
### Model:
```
model:
    modules:
        modules_0:
            model_class: model.features.wav2vec:featureExtractor
            save: True
            freeze: False
            inputs: ['wav#wav.scp']
            arch:
                conf:
                    cp_path: pretrained_models/wav2vec_small.pt
                    device: cuda
            output: ['wav2vec']

        modules_1:
            model_class: model.UTMOS.model:Model
            save: True
            freeze: False
            inputs: ['wav2vec', 'text.listchar', 'ref.listchar', 'domain.emb', 'judge_id.emb', 'score_norm.txt']
            arch:
                conf:
                    hidden_dim: 256
                    emb_dim: 256
                    out_dim: 256
                    n_lstm_layers: 3
                    vocab_size: 2048
                    n_domains: 20
                    domain_dim: 128

                    judge_dim: 128
                    num_judges: 3500
                    projection_hidden_dim: 2048
                    range_clipping: False
                    pref_mode: pref4

            output: ['gt_pref_score', 'utt_pref_score', '_ids']
```
The configure can contains multiple modules. 
`modules_0` loads in wav2vec model and finetune it by `freeze: False`. The input is `wav#wav.scp` which will be explained afterward and the outputs is 'wav2vec'.
`modules_1` is the UTMOS model. The input includes the 'wav2vec' extracted from `modules_0` and other 5 features. It will output 'gt_pref_score', 'utt_pref_score', '_ids'.

### Dataset
```
dataset:
    data_cache: False
    features: ['wav#wav.scp', 'text.listchar', 'ref.listchar', 'domain.emb', 'judge_id.emb', 'score_norm.txt']
    label: []
    fs: 16000

    collate_fns:
        _default: 'repetitive_to_max'
        text.listchar: 'pad_to_max'
        ref.listchar: 'pad_to_max'

    batch_size:
        _default: 3
        dev: 1
        test: 1

    pairing:
        additional_data: 1
        limits: 'same:judge_id.emb'

    data_augmentation:
        wav#wav.scp:
            -
                augment: model.augment.SliceWav:SliceWav
                conf:
                    max_wav_seconds: 10
                aug_when_inferring: False

# data_cache: if the features from wav.scp are saved in RAM or read them online
# features: We will load the wav.scp, text.listchar, ref.listchar, domain.emb, judge_id.emb, score_norm.txt from data/{your_dataset}. (each file extension means different way to read, we will explain them afterwards)
# collate_fns: collate_fns for Pytorch DataLoader. We provide 'pad_to_max', 'repetitive_to_max', 'crop_to_min', 'crop_to_min_rand', 'pass_unique'
# batch_size: training, development, testing batch size
# pairing: The model will load {additional_data} have the same limits. In this configuration, 1 another paired with limit that have the same judge_id in judge_id.emb will be paired for training/development/testing.
# data_augmentation: Data augmentation is applied for wav.scp
```
### Training Configuration
```
optimizer:
    type: torch.optim:Adam
    conf:
        lr: 0.00002

scheduler:
    type: transformers:get_linear_schedule_with_warmup
    conf:
        num_warmup_steps: 4000
        num_training_steps: 15000

trainer:
    iteration_type: 'iteration'
    iterations: 60000
    save_per_iterations: 20000
    eval_per_iterations: 20000
    accum_grad: 4
    loss:
        ClippedMSELoss:
            criterion_class: model.nn.losses:batchedMSELoss
            inputs: ['utt_pref_score', 'gt_pref_score']
            weight: 1

logger:
    log_metrics: True
    logger_conf: conf/UTMOS/logger_score_pref.yaml
    save: ['pred_utt_score.txt#utt_pref_score', 'gt_score.txt#gt_pref_score']

# set optimizer, scheduler here.
# the model is trained 60000 iterations, the model is saved every 20000 iterations and is evaluated on development set every 20000 iterations.
# the model accumulate 4 gradients for each update.
# The model is trained using model.nn.losses:batchedMSELoss loss between 'utt_pref_score' and 'gt_pref_score', which are the output of the modules.
# The logger will save 'utt_pref_score' in {outdir}/save/{data}/{saved_iteration}/pred_utt_score.txt and save 'gt_pref_score' in {outdir}/save/{data}/{saved_iteration}/pred_utt_score.txt
```
## Data
Refer to dataset.py.
We have different file extensions to indicate which way we need to read the file. You can put files in data/{dataset_name}/ for easy joint training, feature selection and subset selection.
1. *.txt

Each line contains lines `{id} {float}`, meaning `id` have `float`

2. *.str

Each line contains lines `{id} {string}`. We convert the string into ascii list using `ord` and save as `torch.tensor`.

3. *.emb

The file contains lines `{id} {emb}`. Each `{emb}` will be registered with a unique `id` for the use like extracting embedding from `torch.nn.Embedding`. 
Noting that `{emb}` will be registered in {outdir}/save/data_ids and used for all the datasets afterwards.

4. *.list

The file contains lines `{id} {float_1} {float_2} ... {float_n}, meaning `id` have a `float` list.

5. *.listemb

The file contains lines `{id} {emb_1} {emb_2} ... {emb_n}`. Each `emb` is split by a space. meaning `id` have a `emb` list. 
Noting that `{emb}` will be registered in {outdir}/save/data_ids and used for all the datasets afterwards.

6. *.listchar

The file contains lines `{id} {char_1}{char_2}...{char_n}`, meaning `id` have a `char` list. 
Note that `{char}`s are not split by spaces, meaning a space can be used as an input like normal text input.

7. wav.scp
   
The core of the dataset.
The file contains lines `{id} {filepath}`, meaning `id` have a `{filepath}`.
We can add prefix `spectrogram#` to extract features while reading or `wav#` to use the raw waveform as input.

8. datasetids

The file contains lines `{id}`, meaning only use the set of `{id}` for reading.

9. trials

The file contains lines `{id_1} {id_2} ... {id_n}`. meaning use pairs of `ids` for reading.

