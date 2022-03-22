from .base import baseExtractor
class extractor(baseExtractor):
    def get_inputs_nums(self):
        raise 1
    def get_feature_name(self):
        raise 'argmax'

    def forward(self, x):
        return x.argmax(dim=0)
