import torch
import librosa
import numpy as np

class wav16k:
    def __call__(self, x):
        f, sr = librosa.load(x, sr=16000)
        return f
    
class pt_loader:
    def __call__(self, x):
        return torch.load(x)
    
class np_loader:
    def __call__(self, x):
        return np.load(x)