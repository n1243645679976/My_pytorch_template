import torch
import fairseq
from .base import baseExtractor

class featureExtractor(baseExtractor):
    def __init__(self, conf):
        super(featureExtractor, self).__init__(conf)
        self.model = fairseq.checkpoint_utils.load_model_ensemble_and_task([conf['cp_path']])[0][0].to(conf['device'])
        self.model.remove_pretraining_modules()
        
    def get_inputs_nums(self):
        return 1

    def get_feature_name(self):
        return 'wav2vec'

    def extract(self, x):
        # waveform
        assert len(x.shape) == 1, f'{x.shape}'
        x = torch.from_numpy(x).to(self.conf['device'])
        x = x.reshape(1, -1)
        with torch.no_grad():
            return self.model(x, mask=False, features_only=True)['x'].squeeze(0).to('cpu')

    def forward(self, x):
        x = x[0].data
        # Batch * Waveform
        return [self.model(x, mask=False, features_only=True)['x']]

    def inference(self, x):
        return self(x)

    def get_output_dim(self):
        return 768

