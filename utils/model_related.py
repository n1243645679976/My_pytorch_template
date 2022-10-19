import torch
import sys
import os
sys.path.append(os.getcwd())
import yaml
from utils.dynamic_import import dynamic_import
from dataset import packed_batch
class defaultModuleLoader():
    def set_pretrained_model(modules, key, pretrained_model):
        module = torch.load(pretrained_model)
        modules[key].load_state_dict(module['model'][key])

class model(torch.nn.Module):
    def __init__(self, conf):
        super(model, self).__init__()
        model_conf = conf['model']
        modules_conf = model_conf['modules']
        self.moduledict = torch.nn.ModuleDict()
        self.update_modules = torch.nn.ModuleList()
        self.module_inputs = {}
        self.module_outputs = {}
        self.input_adaptor = {}
        self.modules_conf = modules_conf
        self.module_keys = []
        for key in modules_conf.keys():
            module_conf = modules_conf[key]

            module_class = dynamic_import(module_conf['model_class'])
            module_class_conf = module_conf.get('arch', {})
            self.moduledict[key] = module_class(**module_class_conf)

            # register output
            self.module_outputs[key] = module_conf['output']

            # register modules for training
            if not module_conf['freeze']:
                self.update_modules.append(self.moduledict[key])

            # resuming training or init with specific initial model
            self.module_inputs[key] = module_conf.get('inputs', [])
            self.module_keys.append(key)
        self.len_dataset_inputs = len(conf['dataset']['features']) - len(conf['dataset'].get('label', []))

    def parameters(self):
        return self.update_modules.parameters()

    def forward(self, x, inference=False):
        packed_data = x
        for module_key in self.module_keys:
            inputs = []
            if self.module_inputs[module_key]:
                for module_input in self.module_inputs[module_key]:
                    inputs.append(x[module_input])
            else:
                for i in range(self.len_dataset_inputs):
                    inputs.append(x[f'_dataset_feat_x{i}'])
                    
            if inference:
                outputs = self.moduledict[module_key].inference(inputs)
            else:
                outputs = self.moduledict[module_key](inputs)
            
            for output_name, module_output in zip(self.module_outputs[module_key], outputs):
                assert output_name not in packed_data
                if isinstance(module_output, packed_batch):
                    packed_data[output_name] = module_output
                else:
                    packed_data[output_name] = packed_batch(module_output)

        return packed_data

    def inference(self, x):
        with torch.no_grad():
            return self(x, inference=True)

    def load_model(self, resume):
        for key in self.modules_conf.keys():
            if not self.modules_conf[key]['save']:
                continue
            if resume:
                self.moduledict[key].load_state_dict(resume[key])
            else:
                initial_model_filename = self.modules_conf.get('initial_model', None)
                if initial_model_filename:
                    initial_model = torch.load(initial_model_filename)
                    keys2model = self.modules_conf.get('keys2model', ['model'])
                    for key in keys2model:
                        initial_model = initial_model[key]
                    self.moduledict[key].load_state_dict(initial_model)
        

def get_model(conf, resume, device):
    print(model)
    _model = model(conf)
    _model.load_model(resume)
    _model = _model.to(device)
    _model.train()

    return _model

def save_model(trainer):
    os.makedirs(os.path.join(trainer.args.exp, 'result'), exist_ok=True)
    model = {
        'iters': trainer.iters,
        'args': trainer.args,
        'conf': trainer.conf,
        'model': {},
        'optimizer': trainer.optimizer.state_dict()
    }
    for module_key in trainer.model.module_keys:
        if trainer.model.modules_conf[module_key]['save']:
            model['model'][module_key] = trainer.model.moduledict[module_key].state_dict()
    torch.save(model, os.path.join(trainer.exp, 'result', f'snapshot.{trainer.iters}'))

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.getcwd())
    with open('conf/svsnet_v2_svs20.yaml') as f:
        conf = yaml.safe_load(f)
    r = model(conf)
    r.args.exp = './test'
    save_model()
