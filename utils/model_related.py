from email.policy import default
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
        outputs = set()
        self.module_keys = []
        for key in modules_conf.keys():
            module_conf = modules_conf[key]

            # set module
            module_class = dynamic_import(module_conf['model_class'])
            module_class_conf = module_conf.get('arch', {})
            module = module_class(**module_class_conf)
            self.moduledict[key] = module

            # register output
            self.module_outputs[key] = module_conf['output']
            for output in self.module_outputs:
                assert output not in outputs, 'replicated output found'
                outputs.add(output)

            # register modules for training
            if not module_conf['freeze']:
                self.update_modules.append(self.moduledict[key])

            # resuming training or init with specific initial model

            # input_adaptor
            input_adaptor_name = module_conf.get('input_adaptor', None)
            input_adaptor_class = dynamic_import(input_adaptor_name) if input_adaptor_name != None else torch.nn.Identity
            self.input_adaptor[key] = input_adaptor_class()

            self.module_inputs[key] = module_conf.get('inputs', [])
            self.module_keys.append(key)
        self.len_dataset_inputs = len(conf['dataset']['features']) - len(conf['dataset']['label'])

    def parameters(self):
        return self.update_modules.parameters()

    def forward(self, x):
        packed_data = x
        for module_key in self.module_keys:
            inputs = []
            if self.module_inputs[module_key]:
                for module_input in self.module_inputs:
                    inputs.append(x[module_input])
            else:
                for i in range(self.len_dataset_inputs):
                    inputs.append(x[f'_dataset_feat_x{i}'])

            adapted_inputs = self.input_adaptor[module_key](inputs)
            outputs = self.moduledict[module_key](adapted_inputs)

            for output_name, module_output in zip(self.module_outputs[module_key], outputs):
                assert output_name not in packed_data
                if isinstance(module_output, packed_batch):
                    packed_data[output_name] = module_output
                else:
                    packed_data[output_name] = packed_batch(module_output)
        return packed_data

    def inference(self, x):
        with torch.no_grad():
            packed_data = self(x)
        return packed_data

    def load_model(self, resume):
        for key in self.modules_conf.keys():
            if resume:
                printed_set = set()
                printed_set1 = set()
                import copy
                q = {n:copy.deepcopy(p) for n, p in self.moduledict[key].named_parameters()}
                self.moduledict[key].load_state_dict(resume[key])
                k = torch.load('exp/train_svsnet_not_mean_svsnet_wavlm/result/snapshot.1')['model']
                
                #for n, p in self.moduledict[key].named_parameters():
                #    if n == 'wavenet.0.first_conv.bias':
                #        print(n, p)#torch.sum((q[n] != p).long()))
            else:
                initial_model_filename = self.modules_conf.get('initial_model', None)
                if initial_model_filename:
                    initial_model_adaptor_class = self.modules_conf.get('initial_model_adaptor', defaultModuleLoader)
                    initial_model_adaptor = initial_model_adaptor_class()
                    initial_model_adaptor.set_pretrained_model(self.moduledict, key, initial_model_filename)
        

def get_model(conf, resume, device):
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
