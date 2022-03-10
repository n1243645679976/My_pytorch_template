import torch
from .base import baseExtractor
from scipy.stats import spearmanr as srcc
class extractor(baseExtractor):
    def get_inputs_nums(self):
        return 2
    def get_feature_name(self):
        return 'srcc'
    def extract(self, x, y):
        return srcc(x, y)[0]
    def forward(self, x, y):
        x = x.reshape(-1)
        y = y.reshape(-1)
        return torch.tensor(self.extract(x, y))