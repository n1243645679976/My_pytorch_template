import torch
from .base import baseExtractor
def noise_scaling(speech, noise, snr, eps=1e-10):
    snr_exp = 10.0 ** (snr / 10.0)
    speech_power = speech.pow(2).sum(dim=-1, keepdim=True)
    noise_power = noise.pow(2).sum(dim=-1, keepdim=True)
    scalar = (speech_power / (snr_exp * noise_power + eps)).pow(0.5)
    scaled_noise = scalar * noise
    return scaled_noise

class extractor(baseExtractor):
    def __init__(self):
        noiselist = '/nas01/homes/cheng22-1000061/MOSNet/UTMOS22/strong/noise.txt'
        self.noiselist = []
        with open(noiselist) as f:
            for line in f.readlines():
                self.noiselist.append(os.path.join('/nas01/homes/cheng22-1000061/dataset/', line.strip()))
        
    def get_inputs_nums(self):
        return 1

    def get_feature_name(self):
        return 'wav'

    def extract(self, x):
        q = random.choose(self.noiselist)
        if len(x) > :
            start_point = np.random.randint(x.shape[1] - max_len)
            x = x[start_point:start_point+max_len]
        amount_to_pad = max_len - x.shape[1]
        padded_x = torch.nn.functional.pad(
            x, (0, amount_to_pad), "constant", 0)
        noise = noise_scaling(x, q, snr)
        x += noise

            for wav, sp in zip(noises):
                if wav.shape[1] > max_len:
                    start_point = np.random.randint(wav.shape[1] - max_len)
                    wav = wav[:, start_point:start_point+max_len]
                amount_to_pad = max_len - wav.shape[1]
                padded_wav = torch.nn.functional.pad(
                    wav, (0, amount_to_pad), "constant", 0)
                output_noises.append(padded_wav)
        return x

    def forward(self,x):
        return self.extract(x)

    def get_output_dim(self):
        return 1
