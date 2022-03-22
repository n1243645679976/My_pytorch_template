import torch
import librosa
import scipy.signal
import numpy as np
from espnet.transform.spectrogram import logmelspectrogram

from .base import baseExtractor
class extractor(baseExtractor):
    def __init__(self, conf):
        super(extractor, self).__init__(conf)
    def get_inputs_nums(self):
        return 1
    def get_feature_name(self):
        return 'fbank'
    def extract(self, x):
        lmspc = logmelspectrogram(
                x=x,
                fs=self.conf['fs'],
                n_mels=self.conf['n_mels'],
                n_fft=self.conf['n_fft'],
                n_shift=self.conf['n_shift'],
                win_length=self.conf['win_length'],
                window=self.conf['window'],
                fmin=self.conf['fmin'],
                fmax=self.conf['fmax'],
            )
        return lmspc.astype(np.float32)
    def forward(self, x):
        return torch.tensor(self.extract(x))
            