fs=16000
stage=0
stop_stage=100

train=train_history
dev=dev_history
test=test_history
features=features
feat_conf=conf/espnet/feat_extract_v2.yaml
train_conf=conf/espnet/tts/tacotron2.yaml
eval_conf=conf/espnet/tts/eval_tacotron2.yaml

extract_feature_online="True"
resume=""
token_type="char"
eos="True"
space="False"

start_testing=0
end_testing=10000
. ./utils/parse_options.sh

set -u
set -x

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: make tokenid"
    command=""
    if [ "$eos" == "True" ]; then command="$command --eos"; fi
    if [ "$space" == "True" ]; then command="$command --space"; fi
    python utils/token2tokenid.py --token data/${train}/text --tokenid data/${train}/tokenid.list --save_diction_path data/$train/diction --token_type "$token_type" $command
    python utils/token2tokenid.py --token data/${dev}/text --tokenid data/${dev}/tokenid.list --diction_path data/$train/diction --token_type "$token_type" $command
    python utils/token2tokenid.py --token data/${test}/text --tokenid data/${test}/tokenid.list --diction_path data/$train/diction --token_type "$token_type" $command
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: extract features"
    if [ $extract_feature_online == "True" ]; then
        echo "Since extract_feature_online set to True, skip extracting features"
    else
        for name in $train $dev $test; do 
                python extract_features.py --outdir $features \
                                           --set $name \
                                           --conf $feat_conf 
        done
    fi
fi

expdir=exp/${train}_`basename ${train_conf%%.*}`
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "stage 1: training tacotron2"
    python main.py --train $train --dev $dev  --conf $train_conf --features $features --exp $expdir --device cuda --resume "$resume" --extract_feature_online $extract_feature_online
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "stage 2: testing tacotron2"
    python test.py --test $test --conf $eval_conf --exp $expdir --start $start_testing --end $end_testing --features $features --device cuda --extract_feature_online $extract_feature_online
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "stage 3: decode tacotron2"
    voc_stats=../pwg/cospro_hifi16k/stats.h5
    voc_conf=../pwg/cospro_hifi16k/config.yml
    voc_checkpoint=../pwg/cospro_hifi16k/checkpoint-900000steps.pkl
    for out_dirs in `ls $expdir/save/$test/`; do 
        [ -d ./take_out/hdf5 ] && rm -r ./take_out/hdf5
        parallel-wavegan-normalize --skip-wav-copy --config "${voc_conf}" --stats "${voc_stats}" --feats-scp $expdir/save/$test/$out_dirs/generate_speech.scp --dumpdir ./take_out/hdf5 --verbose 1
        parallel-wavegan-decode --dumpdir ./take_out/hdf5 --checkpoint ${voc_checkpoint} --outdir ./take_out/$expdir/$test/$out_dirs/ --verbose 1
    done
fi
