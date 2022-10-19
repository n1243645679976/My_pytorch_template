import torch
from .base import baseExtractor

class extractor(baseExtractor):
    def get_inputs_nums(self):
        return 2
    def get_feature_name(self):
        return 'lcc'
    def extract(self, x, y):
        return torch.mean((x == y).float())
    def forward(self, x, y):
        return torch.mean((x == y).float())
    def get_output_dim(self):
        return 1