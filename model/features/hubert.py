import torch
from .base import baseExtractor
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