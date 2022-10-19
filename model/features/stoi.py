import torch
from pystoi import stoi
from .base import baseExtractor
class extractor(baseExtractor):
    def get_inputs_nums(self):
        return 2
    def get_feature_name(self):
        return 'stoi'
    def extract(self, x, y):
        if abs(x.shape[0] - y.shape[0]) / self.conf['fs'] > 0.15:
            print('more than 0.15 second length difference between trial inputs')
        if x.shape[0] > y.shape[0]:
            x_start = (x.shape[0] - y.shape[0]) // 2
            x_end = x_start + y.shape[0]
            x = x[x_start:x_end]
        if y.shape[0] > x.shape[0]:
            y_start = (y.shape[0] - x.shape[0]) // 2
            y_end = y_start + x.shape[0]
            y = y[y_start:y_end]

        stoi_score = stoi(x, y, self.conf['fs'])
        return stoi_score
    def forward(self,x, y):
        return torch.tensor(self.extract(x, y)).reshape(1)
    def get_output_dim(self):
        return 1

