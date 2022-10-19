import torch
import cdpam
from pystoi import stoi
from .base import baseExtractor

class extractor(baseExtractor):
    def __init__(self, conf):
        super(extractor, self).__init__(conf)
        self.cdpam_model = cdpam.CDPAM()
    def get_inputs_nums(self):
        return 2
    def get_feature_name(self):
        return 'cdpam'
    def extract(self, x, y):
        return self.cdpam_model.forward(x, y)
    def forward(self, x, y):
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        return torch.tensor(self.extract(x, y)).reshape(1)
    def get_output_dim(self):
        return 1
