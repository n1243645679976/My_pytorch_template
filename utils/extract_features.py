import os
import sys
import tqdm
import yaml
import torch
import librosa
import fairseq
import argparse
import scipy.signal

import numpy as np

from utils.kaldi_reader import KaldiReader

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', help='output directory')
parser.add_argument('--feat', help='feature to extract')
parser.add_argument('--set', help='dataset to extract feature')
parser.add_argument('--feat_conf', help='config of feature extraction')
args = parser.parse_args()

# feat extractor
class FeatExtractor:
    def __init__(self, conf_file):
        self.window = scipy.signal.hamming
        with open(conf_file) as conf:
            self.conf = yaml.safe_load(conf)

        for key in self.conf.keys():
            if 'hubert'  == key:
                self.hubert_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.conf['cp_path']])
                self.hubert_model = self.hubert_model[0]
                self.hubert_model.eval()
                self.device = torch.device(self.conf['device'])
                self.hubert_model = self.hubert_model.to(self.device)
                
        
    def spectrogram(self, x):
        # input waveform: (1xT)
        # return spectrogram: (TxF)
        linear = librosa.stft(y=x, 
                              n_fft=self.conf['fft_size'],
                              hop_length=self.conf['hop_length'],
                              win_length=self.conf['win_length'],
                              window=self.window)
        linear = linear.T
        linear = np.abs(linear)
        return linear.astype(np.float)
        
    def hubert_feature(self, x):
        x = x.reshape(1, -1)
        F = torch.from_numpy(x).to(self.device).float()
        feature = self.hubert_model(F, features_only=True, mask=False)['x']
        causal = feature.detach().to('cpu').numpy().astype(np.float)
        return causal

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
        F = x.to(self.device).float()
        feature = self.hubert_model(F, features_only=True, mask=False)['x']
        return feature

# main
if __name__ == "__main__":
    expdir = os.path.join(args.outdir, args.set, args.feat)
    os.makedirs(expdir, exist_ok=True)
    assert os.path.isfile(f'data/{args.set}/wav.scp'), f'please prepare file list as data/{args.set}/wav.scp'
    assert os.path.isfile(args.feat_conf), f'{args.feat_conf} doesn\'t exist'

    feat_extractor = FeatExtractor(args.feat_conf)
    if args.feat == 'spectrogram':
        feat_ext = feat_extractor.spectrogram
    elif args.feat == 'hubert':
        feat_ext = feat_extractor.hubert_feature
    else:
        raise NotImplementedError(f'not supported feature {args.feat}')

    for key, value in tqdm.tqdm(KaldiReader(f'scp:data/{args.set}/wav.scp')):
        wav = value[1] / 32768.
        feat = feat_ext(wav)
        torch.save(torch.tensor(feat), os.path.join(expdir, key + '.pt'), _use_new_zipfile_serialization=False)

    with open(os.path.join(expdir, 'command.txt'), 'w+') as f:
        f.write(' '.join(sys.argv))

