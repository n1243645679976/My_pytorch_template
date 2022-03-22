from email.policy import default
import os
import time
import yaml
import sys
import torch
import numpy as np
from collections import defaultdict
from utils.dynamic_import import dynamic_import

"""
ref: https://stackoverflow.com/questions/44904290/getting-duplicate-keys-in-yaml-using-python
"""
from ruamel_yaml import YAML
from ruamel_yaml.constructor import SafeConstructor
def construct_yaml_map(self, node):
    # test if there are duplicate node keys
    data = []
    yield data
    for key_node, value_node in node.value:
        key = self.construct_object(key_node, deep=True)
        val = self.construct_object(value_node, deep=True)
        data.append((key, val))
SafeConstructor.add_constructor(u'tag:yaml.org,2002:map', construct_yaml_map)


class Logger():
    def __init__(self, exp, args, conf, log_name='train'):
        print(f'set {log_name} logger')
        self._accum_loss = 0
        self.exp = exp
        self.log_name = log_name
        os.makedirs(self.exp, exist_ok=True)
        with open(os.path.join(self.exp, 'config.yaml'), 'w+') as outfile:
            yaml.dump(conf, outfile, default_flow_style=False)
        with open(os.path.join(self.exp, 'args'), 'w+') as outfile:
            outfile.write(' '.join(sys.argv))
        self.is_first_line_written = False
        self.record = defaultdict(list)
        self.record_size = defaultdict(list)

        self.save = conf.get('save', [])
        if self.save:
            self.save_dir = os.path.join(self.exp, 'save', self.log_name)
            os.makedirs(self.save_dir, exist_ok=True)
            self.save_record = defaultdict(list)

        self.log_metrics = conf['log_metrics']
        if conf['log_metrics']:
            with open(conf['logger_conf']) as f:
                metric_conf_str = f.read()
            _yaml = YAML(typ='safe')
            self.metric_conf = _yaml.load(metric_conf_str) 
            self.metric_record = defaultdict(list)
            self.needed_inputs = set()
            metric_package = 'model.features.{}:extractor'
            aggregate_package = 'utils.aggregation:{}'
            self.inputs = defaultdict(lambda: defaultdict(list))
            self.inputs_functions = defaultdict(list)
            self.aggregate_functions = defaultdict(list)

            def dfs(command_conf):
                if isinstance(command_conf[1], list):
                    # diction, command_conf[*][0] is key, command_conf[*][1] is the value
                    if isinstance(command_conf[1][0], tuple):
                        for command in command_conf[1]:
                            dfs(command)
                    # list, command_conf[*] is the leaf
                    else:
                        for command in command_conf[1]:
                            self.needed_inputs.add(command)
                # else, it may be string
                else:
                    self.needed_inputs.add(command_conf[1])
                
                aggregate_func, metric_func = command_conf[0].split('.')
                if aggregate_func not in self.aggregate_functions:
                    self.aggregate_functions[aggregate_func] = dynamic_import(aggregate_package.format(aggregate_func))
                if metric_func not in self.inputs_functions:
                    self.inputs_functions[metric_func] = dynamic_import(metric_package.format(metric_func))({})

            for metric_conf in self.metric_conf:
                dfs(metric_conf[1][0])

    def iter_metric_dict(self, conf):
        # assert isinstance(conf, tuple)
        #print(conf, '\n'*10)
        all_inputs = []
        outputs_dict = {}
        aggregate_func, metric_func = conf[0].split('.')
        if isinstance(conf[1], list):
            if isinstance(conf[1][0], tuple):
                # diction, command_conf[*][0] is key, command_conf[*][1] is the value
                for _conf in conf[1]:
                    inputs = self.iter_metric_dict(_conf)
                    all_inputs.append(inputs)
            else:
                # list, command_conf[*] is the leaf
                for q in conf[1]:
                    all_inputs.append(self.metric_record[q])
        else:
            all_inputs.append(self.metric_record[conf[1]])
        
        # all_inputs: [[(id1, feat1[1]), (id2, feat1[2]), ...], [(id1, feat2[1]), (id2, feat2[2]), ...], ...]
        input_list = []
        inputs_dict = defaultdict(lambda :defaultdict(list))
        for i, _inputs in enumerate(all_inputs):
            for _id, _input in _inputs:
                inputs_dict[i][self.aggregate_functions[aggregate_func](_id)].append(_input)
            input_list.append([])
        for input_key in inputs_dict[0].keys():
            for i in inputs_dict.keys():
                input_list[i] = torch.cat(inputs_dict[i][input_key], dim=0)
            outputs_dict[input_key] = self.inputs_functions[metric_func](*input_list).reshape(-1)
        
        outputs = [(id, value) for id, value in outputs_dict.items()]
        return outputs

    def log_and_clear_record(self, iter):
        if self.log_metrics:
            for metric_conf in self.metric_conf:
                self.record[metric_conf[0]] = self.iter_metric_dict(metric_conf[1][0])[0][1].numpy()
                self.record_size[metric_conf[0]] = [1]
            self.metric_record = defaultdict(list)

        if self.save:
            for save_command in self.save:
                file_name, data_name = save_command.split('#')
                data = self.save_record[data_name]
                save_dir = os.path.join(self.save_dir, str(iter))
                os.makedirs(save_dir, exist_ok=True)
                # handle each file extension here
                if file_name.endswith('.txt'):
                    with open(os.path.join(save_dir, file_name), 'w+') as w:
                        for key, value in sorted(data):
                            output = ' '.join(map(str, value.reshape(-1)))
                            w.write(f'{key} {output}\n')
                elif file_name == 'pt':
                    # do things with data
                    for key, value in sorted(data):
                        torch.save(value, os.path.join(save_dir, key + '.pt'))
                elif file_name == 'scp':
                    import kaldiio
                    save_data = {}
                    for key, value in sorted(data):
                        save_data[key] = value
                    kaldiio.save_ark(os.path.join(save_dir, file_name + '.ark'), save_data, scp=os.path.join(save_dir, data_name + '.scp'))
                elif file_name.endswith('.wav'):
                    # do things with data
                    pass
                elif file_name.endswith('.png'): # something like attention map
                    # do things with data
                    pass 
            
            self.save_record = defaultdict(list)

        record_keys = list(self.record.keys())
        if not self.is_first_line_written:
            with open(os.path.join(self.exp, self.log_name + '.log'), 'w+') as outfile:
                outfile.write('iter\t')
                outfile.write('\t'.join(record_keys))
                outfile.write('\n')
            self.is_first_line_written = True
        with open(os.path.join(self.exp, self.log_name + '.log'), 'a+') as outfile:
            outfile.write(f'{iter}')
            print(f'{self.log_name}: {iter}', end='')
            for key in record_keys:
                outfile.write(f'\t{np.sum(self.record[key]) / np.sum(self.record_size[key]):.4f}')
                print(f'\t{np.sum(self.record[key]) / np.sum(self.record_size[key]):.4f}', end='')
            outfile.write('\n')
            print('')
        self.record = defaultdict(list)
        self.record_size = defaultdict(list)
            
    def register_one_record(self, packed_data, size):
        if self.log_metrics:
            for key in self.needed_inputs:
                if packed_data[key].len != None:
                    self.metric_record[key].extend([(id, each_batch[:_len].detach().cpu()) for id, each_batch, _len in zip(packed_data['_ids'], packed_data[key].data, packed_data[key].len)])
                else:
                    self.metric_record[key].extend([(id, each_batch.detach().cpu()) for id, each_batch in zip(packed_data['_ids'], packed_data[key].data)])
        print(self.save)
        if self.save:
            for save_command in self.save:
                file_name, key = save_command.split('#')
                if packed_data[key].len != None:
                    self.save_record[key].extend([(id, each_batch[:_len].detach().cpu().numpy()) for id, each_batch, _len in zip(packed_data['_ids'], packed_data[key].data, packed_data[key].len)])
                else:
                    self.save_record[key].extend([(id, each_batch.detach().cpu().numpy()) for id, each_batch in zip(packed_data['_ids'], packed_data[key].data)])

        loss = packed_data.get('_loss', None)
        if loss != None:
            loss = loss.data
            for key, value in loss.items():
                self.record[key].append(value.detach().cpu().numpy() * size)
                self.record_size[key].append(size)
    


if __name__ == '__main__':
    from utils.parse_config import get_train_config
    import torch
    args, conf = get_train_config()
    loss1 = {'overall_loss': torch.tensor([2.3]), 'l1_loss': torch.tensor([1.2]), 'l2_loss': torch.tensor([1.1])}
    loss2 = {'overall_loss': torch.tensor([2.4]), 'l1_loss': torch.tensor([1.5]), 'l2_loss': torch.tensor([0.9])}
    bs1 = torch.randn(64, 1).shape[0]
    bs2 = torch.randn(32, 1).shape[0]

    test_logger = Logger(args.exp, args, conf, 'train')
    test_logger.register_one_record(loss1, bs1)
    test_logger.register_one_record(loss2, bs2)
    test_logger.log_and_clear_record(1000)
    test_logger.register_one_record(loss1, bs1)
    test_logger.register_one_record(loss1, bs1)
    test_logger.log_and_clear_record(2000)
    
