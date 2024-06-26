import torch
from .base import baseExtractor
from scipy.stats import pearsonr as lcc
class extractor(baseExtractor):
    def get_inputs_nums(self):
        return 2
    def get_feature_name(self):
        return 'lcc'
    def extract(self, x, y):
        return lcc(x, y)[0]
    def forward(self, x, y):
        x = x.reshape(-1)
        y = y.reshape(-1)
        return torch.tensor(self.extract(x, y))
    def get_output_dim(self):
        return 1