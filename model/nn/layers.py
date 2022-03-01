import torch

def init(paras):
    # init parameters with xavier_uniform weight and zeros bias
    # example: init(model.parameters())
    for name, para in paras:
        try:
            if 'weight' in name:
                # assume using ReLU layer as activation function
                torch.nn.init.xavier_uniform_(para, gain=1.414)
            if 'bias' in name:
                torch.nn.init.zeros_(para)
        except Exception as e:
            pass
    
class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)
        init(self.named_parameters())

class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)
        init(self.named_parameters())

class LSTM(torch.nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
        init(self.named_parameters())

class Linear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        init(self.named_parameters())
