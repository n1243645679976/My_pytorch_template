import abc
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class baseExtractor(torch.nn.Module, abc.ABC):
    def __init__(self, conf):
        super(baseExtractor, self).__init__()
        self.conf = conf
    @abc.abstractmethod
    def get_inputs_nums(self):
        raise NotImplemented
    @abc.abstractmethod
    def get_feature_name(self):
        raise NotImplemented
        
    def get_default_input_filenames(self):
        """
        if wavscp_nums == 1: ['wav.scp']
        else: ['trial', 'wav.scp', 'wav1.scp', 'wav2.scp', ...]
        """
        wavscp_nums = self.get_inputs_nums()
        if wavscp_nums == 1:
            return ['wav.scp']
        inputs = ['trial', 'wav.scp']
        for i in range(1, wavscp_nums):
            inputs.append(f'wav{i}.scp')
        return inputs
