import torch
import librosa
import scipy.signal
import numpy as np
from espnet.transform.spectrogram import logmelspectrogram
from resemblyzer import VoiceEncoder, preprocess_wav

from .base import baseExtractor
class extractor(baseExtractor):
    def __init__(self, conf):
        super(extractor, self).__init__(conf)
        self.encoder = VoiceEncoder()
    def get_inputs_nums(self):
        return 1
    def get_feature_name(self):
        return 'spkemb'
    def extract(self, x):
        """
        return 256-dim value
        """
        wav = preprocess_wav(x, self.conf['fs'])
        embed = self.encoder.embed_utterance(wav).reshape(-1)
        return embed
    def forward(self, x):
        return torch.tensor(self.extract(x))
            