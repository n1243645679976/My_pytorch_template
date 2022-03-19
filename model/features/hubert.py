import torch
from .base import baseExtractor
import fairseq
class extractor(baseExtractor):
    def __init__(self, conf):
        super(extractor, self).__init__(conf)
        self.hubert_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.conf['hubert']['cp_path']])
        self.hubert_model = self.hubert_model[0]
        self.hubert_model.eval()
        self.hubert_device = torch.device(self.conf['hubert']['device'])
        self.hubert_model = self.hubert_model.to(self.hubert_device)
    def get_inputs_nums(self):
        return 1
    def get_feature_name(self):
        return 'hubert'
    def forward(self, x):
        x = x.reshape(1, -1)
        F = x.to(self.hubert_device)
        feature = self.hubert_model(F, features_only=True, mask=False)['x']
        return feature
    def extract(self, x):
        x = x.reshape(1, -1)
        F = torch.from_numpy(x).to(self.hubert_device).float()
        feature = self.hubert_model(F, features_only=True, mask=False)['x']
        causal = feature.detach().to('cpu').numpy().astype(np.float32)
        return causal


if __name__ == "__main__":
    wav_input_16khz = torch.randn(1,10320)
    ext = extractor({})
    rep = ext(wav_input_16khz)
    print(rep.shape, wav_input_16khz.shape)

    len_wavs = [5000, 6000, 7000]
    wav_input_16khz = [torch.randn(l) for l in len_wavs]
    x = pad_sequence(wav_input_16khz, batch_first=True)
    x = pack_padded_sequence(x, len_wavs, batch_first=True, enforce_sorted=False)
    rep = ext(x)
    print(rep)