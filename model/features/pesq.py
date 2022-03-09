import torch
from pesq import pesq
from .base import baseExtractor
class extractor(baseExtractor):
    def get_inputs_nums(self):
        return 2
    def get_feature_name(self):
        return 'pesq'
    def extract(self, x, y):
        pesq_score = pesq(self.conf['fs'], x, y, mode=self.conf['mode'])
        return pesq_score
    def forward(self, x, y):
        return torch.tensor(self.extract(x, y)).reshape(1)

