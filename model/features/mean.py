import torch
from .base import baseExtractor
class extractor(baseExtractor):
    def get_inputs_nums(self):
        return 1
    def get_feature_name(self):
        return 'lcc'
    def extract(self, x):
        return np.mean(x)
    def forward(self, x):
        return torch.mean(x.float())
    def get_output_dim(self):
        return 1

