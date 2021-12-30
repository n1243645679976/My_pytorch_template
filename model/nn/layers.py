import torch

def init(paras):
    # init parameters with xavier_uniform weight and zeros bias
    # example: init(model.parameters())
    for name, para in paras:
        try:
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(para, gain=1.414)
            if 'bias' in name:
                torch.nn.init.zeros_(para)
        except Exception as e:
            pass
    
class Conv2d(torch.nn.Conv2d):
    def __init__(self, **kwargs):
        super(Conv2d, self).__init__(**kwargs)
        init(self.named_parameters())

class LSTM(torch.nn.LSTM):
    def __init__(self, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        init(self.named_parameters())

class Linear(torch.nn.Linear):
    def __init__(self, **kwargs):
        super(Linear, self).__init__(**kwargs)
        init(self.named_parameters())
