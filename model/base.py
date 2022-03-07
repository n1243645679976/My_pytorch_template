import abc
import torch

class baseExtractor(torch.nn.Module, abc.ABC):
    def __init__(self, conf):
        super(baseExtractor, self).__init__()
        self.conf = conf
    @abc.abstractmethod
    def get_feature_nums(self):
        raise NotImplemented

    def get_default_inputs(self):
        """
        if wavscp_nums == 1: ['wav.scp']
        else: ['trial', 'wav.scp', 'wav1.scp', 'wav2.scp', ...]
        """
        wavscp_nums = self.get_feature_nums()
        inputs = []
        for i in range(wavscp_nums):
            inputs.append(f'_dataset_feat_x{i}')
        return inputs
