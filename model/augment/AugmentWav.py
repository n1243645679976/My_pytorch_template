
import augment
import torch
import random

class ChainRunner:
    """
    Takes an instance of augment.EffectChain and applies it on pytorch tensors.
    """

    def __init__(self, chain):
        self.chain = chain

    def __call__(self, x):
        """
        x: torch.Tensor, (channels, length). Must be placed on CPU.
        """
        src_info = {'channels': x.size(0),  # number of channels
                    'length': x.size(1),   # length of the sequence
                    'precision': 32,       # precision (16, 32 bits)
                    'rate': 16000.0,       # sampling rate
                    'bits_per_sample': 32}  # size of the sample

        target_info = {'channels': 1,
                       'length': x.size(1),
                       'precision': 32,
                       'rate': 16000.0,
                       'bits_per_sample': 32}

        y = self.chain.apply(
            x, src_info=src_info, target_info=target_info)

        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()
        return y


def random_pitch_shift(a=-300, b=300):
    return random.randint(a, b)

def random_time_warp(f=1):
    # time warp range: [1-0.1*f, 1+0.1*f], default is [0.9, 1.1]
    return 1 + f * (random.random() - 0.5) / 5


class AdditionalDataBase():
    def __init__(self, cfg=None) -> None:
        self.cfg = cfg

    def __call__(self, data):
        return self.process_data(data)

    def process_data(self, data):
        raise NotImplementedError

    def collate_fn(self, batch):
        return dict()

class AugmentWav(AdditionalDataBase):
    def __init__(self,pitch_shift_minmax,random_time_warp_f,phase='train', cfg=None):
        super().__init__(cfg)
        self.chain = augment.EffectChain()
        self.chain.pitch(random_pitch_shift(pitch_shift_minmax['min'], pitch_shift_minmax['max'])).rate(16000)
        self.chain.tempo(random_time_warp(random_time_warp_f))
        self.chain = ChainRunner(self.chain)
        self.phase = phase
    def __call__(self, data):
        # input: [Time]
        # output: [Time]
        data = data.reshape(1, -1)
        if self.phase=='train':
            augmented_wav = self.chain(data)
        else:
            augmented_wav = data
        return augmented_wav.squeeze(0)

if __name__ == '__main__':
    a = AugmentWav({'min':-300, 'max':300}, 1.0)
    b = torch.randn(1, 10000)
    print(a(b).shape)