import torch
from .base import baseExtractor
from sklearn.metrics import mean_squared_error as mse
class extractor(baseExtractor):
    def get_inputs_nums(self):
        return 2
    def get_feature_name(self):
        return 'mse'
    def extract(self, x, y):
        return mse(x, y)
    def forward(self, x, y):
        return torch.tensor(self.extract(x, y))

