import torch
from .base import baseExtractor
class extractor(baseExtractor):
    def get_inputs_nums(self):
        return 1
    def get_feature_name(self):
        return 'wav'
    def extract(self, x):
        # we don't process wav raw feature
        return x
    def forward(self,x):
        return torch.tensor(self.extract(x)).reshape(-1)
