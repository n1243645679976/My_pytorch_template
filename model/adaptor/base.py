import yaml
import torch
from utils.dynamic_import import dynamic_import

class baseAdaptor(torch.nn.Module):
    def __init__(self, conf):
        super(baseAdaptor, self).__init__()
        # set module
        module_class = dynamic_import(conf['model_class'])

        module_class_conf = {}
        module_class_conf = conf.get('arch', {})

        self.model = module_class(**module_class_conf)
        
    def forward(self, x):
        return self.model(x)
    def inference(self, x):
        return self.model(x)