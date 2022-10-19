import torch
from .base import baseExtractor

class extractor(baseExtractor):
    def __init__(self, conf):
        super(extractor, self).__init__(conf)
        self.randlen = int(conf['length'])
        if 'dim' in conf:
            self.dim = int(conf['dim'])
        else:
            self.dim = 0
    def get_inputs_nums(self):
        return 1
    def get_feature_name(self):
        return 'crop'
    def extract(self, x):
        return self.forward(x)
    def forward(self, x):
        maxlen = x.shape[self.dim]
        if self.randlen > maxlen:
            raise Exception('length of x is smaller than random_len in crop function')
        start_index = torch.rand(1) * (maxlen - self.randlen + 1)
        if self.dim != 0:
            x = x.transpose(0, self.dim)
        x = x[start_index:start_index+self.randlen]
        if self.dim != 0:
            x = x.transpose(0, self.dim)
        return x
    def get_output_dim(self):
        return 1
