import torch
import librosa
import scipy.signal
import numpy as np
from .base import baseExtractor
class extractor(baseExtractor):
    def __init__(self, conf):
        super(extractor, self).__init__(conf)
        self.window = scipy.signal.windows.hamming
    def get_inputs_nums(self):
        return 1
    def get_feature_name(self):
        return 'spectrogram'
    def extract(self, x):
        linear = librosa.stft(y=x, 
                              n_fft=self.conf['fft_size'],
                              hop_length=self.conf['hop_length'],
                              win_length=self.conf['win_length'],
                              window=self.window)
        linear = linear.T
        linear = np.abs(linear)
        return linear.astype(np.float32)
    def forward(self, x):
        return torch.tensor(self.extract(x))