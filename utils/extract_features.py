from collections import defaultdict
import os
import sys
import tqdm
import torch
import librosa
import fairseq
import soundfile as sf
import scipy.signal

import numpy as np
from utils.parse_config import get_feat_config
from pesq import pesq
from pystoi import stoi
"""
Use wav.scp in data/{set} to extract feature

If you need to derive some feature(e.g. pesq, stoi) by 2 wavs, then write trial and trial.scp.
The first column of trial is the key in wav.scp while the second column of trial is the key in trial.scp.

"""

def read_wavscp(file):
    wavscp = {}
    with open(file) as f:
        for line in f.read().splitlines():
            key, value = line.split()
            wavscp[key] = value
    return wavscp

def get_sound(file, waveforms, conf_fs, adapt_fs):
    # use filepath as key
    if file in waveforms:
        return waveforms[file]
    
    x, fs = librosa.load(file)
    # fs related, check fs and resample
    if fs != conf_fs:
        if adapt_fs:
            if not get_sound.fs_warning:
                get_sound.fs_warning = True
                print(f'fs not the same: {fs} v.s. {conf_fs} on {file}, adapting automatically')
            x = librosa.resample(x, fs, conf_fs)
        else:
            raise Exception(f'fs not the same {fs} v.s. {conf_fs} on {file}, setting adapt_fs to True to resample automatically')
    waveforms[file] = x
    return x
get_sound.fs_warning = False

# feat extractor
class FeatExtractor:
    def __init__(self, args, conf):
        self.conf = conf
        self.window = scipy.signal.windows.hamming
        for feature, output in zip(self.conf['features_single']['features'], self.conf['features_single']['output']):
            if output == 'pt':
                os.makedirs(os.path.join(args.outdir, args.set, feature), exist_ok=True)

        for feature, output in zip(self.conf['features_trials']['features'], self.conf['features_trials']['output']):
            if output == 'pt':
                os.makedirs(os.path.join(args.outdir, args.set, feature), exist_ok=True)

        for key in self.conf['features_single']['features']:
            if 'hubert'  == key:
                self.hubert_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.conf['hubert']['cp_path']])
                self.hubert_model = self.hubert_model[0]
                self.hubert_model.eval()
                self.hubert_device = torch.device(self.conf['hubert']['device'])
                self.hubert_model = self.hubert_model.to(self.hubert_device)
                
    def extract(self, x, feature, y=None):
        if feature == 'spectrogram':
            feat = self.spectrogram(x)
        elif feature == 'hubert':
            feat = self.hubert_feature(x)
        elif feature == 'stoi':
            feat = self.stoi(x, y)
        elif feature == 'pesq':
            feat = self.pesq(x, y)
        else:
            raise NotImplementedError(f'not supported feature {feature}')
        return feat

    def spectrogram(self, x):
        # input waveform: (1xT)
        # return spectrogram: (TxF)
        linear = librosa.stft(y=x, 
                              n_fft=self.conf['spectrogram']['fft_size'],
                              hop_length=self.conf['spectrogram']['hop_length'],
                              win_length=self.conf['spectrogram']['win_length'],
                              window=self.window)
        linear = linear.T
        linear = np.abs(linear)
        return linear.astype(np.float32)
        
    def hubert_feature(self, x):
        x = x.reshape(1, -1)
        F = torch.from_numpy(x).to(self.hubert_device).float()
        feature = self.hubert_model(F, features_only=True, mask=False)['x']
        causal = feature.detach().to('cpu').numpy().astype(np.float32)
        return causal

    def pesq(self, x, y):
        pesq_score = pesq(self.conf['fs'], x, y, mode=self.conf['pesq']['mode'])
        return pesq_score

    def stoi(self, x, y):
        if abs(x.shape[0] - y.shape[0]) / self.conf['fs'] > 0.15:
            print('more than 0.15 second length difference between trial inputs')
        if x.shape[0] > y.shape[0]:
            x_start = (x.shape[0] - y.shape[0]) // 2
            x_end = x_start + y.shape[0]
            x = x[x_start:x_end]
        if y.shape[0] > x.shape[0]:
            y_start = (y.shape[0] - x.shape[0]) // 2
            y_end = y_start + x.shape[0]
            y = y[y_start:y_end]

        stoi_score = stoi(x, y, self.conf['fs'])
        return stoi_score

# main
if __name__ == "__main__":
    args, conf = get_feat_config()
    print(f'start extracting features in {args.conf}')

    feat_extractor = FeatExtractor(args, conf)
    
    wavscp = read_wavscp(f'data/{args.set}/wav.scp')
    outputs = defaultdict(lambda :defaultdict(list)) # feature -> key -> feat
    waveforms = {}
    print('extracting features from wav.scp')
    for key, value in tqdm.tqdm(wavscp.items()):
        x = get_sound(value, waveforms, conf['fs'], conf['adapt_fs'])
        for feature, output in zip(conf['features_single']['features'], conf['features_single']['output']):
            outputs[feature]['!@#$ output $#@!'] = output # FIXME: dirty way to save output type
            feat = feat_extractor.extract(x, feature)
            if output == 'pt':
                torch.save(torch.tensor(feat), os.path.join(args.outdir, args.set, feature, key + '.pt'), _use_new_zipfile_serialization=False)
            elif output.endswith('txt'):
                outputs[feature][key].append(feat)
            else:
                raise Exception(f'not implemented output: {output}')
    
    if len(conf['features_trials']) > 0:
        print('extracting features from trial')
        trial_outputs = defaultdict(lambda :defaultdict(list))
        trialscp = read_wavscp(f'data/{args.set}/trial.scp')
        with open(f'data/{args.set}/trial') as f:
            for line in tqdm.tqdm(f.read().splitlines()):
                key1, key2 = line.split()
                v1 = get_sound(wavscp[key1], waveforms, conf['fs'], conf['adapt_fs'])
                v2 = get_sound(trialscp[key2], waveforms, conf['fs'], conf['adapt_fs'])
                for feature, output in zip(conf['features_trials']['features'], conf['features_trials']['output']):
                    trial_outputs[feature]['!@#$ output $#@!'] = output # FIXME: dirty way to save output type
                    feat = feat_extractor.extract(x=v1, y=v2, feature=feature)
                    if output.split('.')[0] == 'trial':
                        key = key1 + '|' + key2
                    else:
                        key = key1
                    trial_outputs[feature][key].append(feat)
    
    for feature in trial_outputs.keys():
        output = trial_outputs[feature]['!@#$ output $#@!']
        func_type, output_type = output.split('.')
        if func_type == 'mean':
            aggregate_func = lambda x, y: [(x, np.mean(y))]
        elif func_type == 'min':
            aggregate_func = lambda x, y: [(x, np.min(y))]
        elif func_type == 'max':
            aggregate_func = lambda x, y: [(x, np.max(y))]
        elif func_type == 'trial':
            aggregate_func = lambda x, y: [(x, _y) for _y in y]
        else:
            raise NotImplementedError(f'not implemented aggregate {aggregate_func}')

        if output_type == 'txt':
            with open(f'data/{args.set}/{feature}.{output_type}', 'w+') as w:
                for key in trial_outputs[feature]:
                    if key == '!@#$ output $#@!':
                        continue
                    for key, y in aggregate_func(key, trial_outputs[feature][key]):
                        w.write(f'{key} {y}\n')
        else:
            raise NotImplementedError(f'not implemented output_type: {output_type}')

    with open(os.path.join(args.outdir, args.set, 'command.txt'), 'w+') as w:
        w.write(' '.join(sys.argv))


# In progress
class onlineFeatureExtractor:
    def __init__(self, conf):
        self.online_window = torch.hamming_window()
        self.preprocessing_list = conf['preprocess']
        self.feature_list = conf['feature']

    def spectrogram(self, x):
        linear = torch.stft(y=x,
                            n_fft=self.conf['fft_size'],
                            hop_length=self.conf['hop_length'],
                            win_length=self.conf['win_length'],
                            window=self.online_window,
                            return_complex=False)
        linear = torch.sqrt((linear ** 2).sum(-1))
        return linear

    def hubert_feature(self, x):
        x = x.reshape(1, -1)
        F = x.to(self.hubert_device).float()
        feature = self.hubert_model(F, features_only=True, mask=False)['x']
        return feature
