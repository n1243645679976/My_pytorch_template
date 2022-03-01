import os
import time
import yaml
import sys
import numpy as np
from collections import defaultdict
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

    def log_and_clear_record(self, iter):
        if not self.is_first_line_written:
            with open(os.path.join(self.exp, self.log_name + '.log'), 'w+') as outfile:
                outfile.write('iter\t')
                outfile.write('\t'.join(sorted(self.record.keys())))
                outfile.write('\n')
            self.is_first_line_written = True
        with open(os.path.join(self.exp, self.log_name + '.log'), 'a+') as outfile:
            outfile.write(f'{iter}')
            print(f'{self.log_name}: {iter}', end='')
            for key in sorted(self.record.keys()):
                outfile.write(f'\t{np.sum(self.record[key]) / np.sum(self.record_size[key]):.4f}')
                print(f'\t{np.sum(self.record[key]) / np.sum(self.record_size[key]):.4f}', end='')
            outfile.write('\n')
            print('')
        self.record = defaultdict(list)
        self.record_size = defaultdict(list)
            
    def register_one_record(self, loss, size):
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
    
