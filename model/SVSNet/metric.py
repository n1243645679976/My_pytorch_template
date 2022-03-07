from collections import defaultdict
import itertools
from scipy.stats import spearmanr
import numpy as np
import torch

def get_sysid_func(dataset):
    def get_sysid(uttid):
        uttid = uttid.split('#')[0]
        utt1, utt2 = uttid.split('&')
        if dataset == 'VCC18':
            sys1 = utt1.split('_')[0] + utt1.split('_')[-1]
            sys2 = utt2.split('_')[0] + utt2.split('_')[-1]
            if not (sys1[0] in 'ST' and sys2[0] in 'ST'):
                sys1, sys2 = max(sys1, sys2), min(sys1, sys2)
            return sys1 + sys2
        elif dataset == 'VCC20':
            sysid = utt1.split('_')[0]
            return sysid
        else:
            raise NotImplementedError(f'not supported dataset {dataset}')
    return get_sysid
    
def get_aggreated_metrics(metric_conf):
    def aggreated_metrics(uttlcc, uttsrcc, uttmse, syslcc, syssrcc, sysmse):
        if metric_conf == 'sys_no_mse':
            return syslcc + syssrcc
        elif metric_conf == 'sys':
            return syslcc + syssrcc - sysmse
        elif metric_conf == 'all_no_mse':
            return syslcc + syssrcc + uttlcc + uttsrcc
        elif metric_conf == 'all':
            return syslcc + syssrcc - sysmse + uttlcc + uttsrcc - uttmse
        else:
            raise NotImplementedError(f'no mosnet metric {metric_conf}')
    return aggreated_metrics
        
def get_metrics(utty, uttpred, sysy, syspred):
    return [np.corrcoef(utty, uttpred)[0][1], spearmanr(utty, uttpred)[0], np.sqrt(np.mean((utty-uttpred)**2)), \
            np.corrcoef(sysy, syspred)[0][1], spearmanr(sysy, syspred)[0], np.sqrt(np.mean((sysy-syspred)**2))]

class SVSNet_metric():
    def __init__(self, args, exp, conf):
        self.get_sysid = get_sysid_func(conf['dataset'])
        self.metric = get_aggreated_metrics(conf['metric'])
        self.metric_type = conf['metric_type']
        self.data = defaultdict(lambda :defaultdict(lambda :defaultdict(list)))
        self.best_record = None
        self.exp = exp

    def write(self, id, y, pred):
        y = y[0].data.cpu().detach().numpy()
        pred = pred[0].cpu().detach().numpy()
        for _id, _y, _pred in zip(id, y, pred):
            sysid = self.get_sysid(_id)
            self.data['utt']['y'][_id].append(_y)
            self.data['utt']['pred'][_id].append(_pred)
            self.data['sys']['y'][sysid].append(_y)
            self.data['sys']['pred'][sysid].append(_pred)
        
    def write_all(self, ep):
        utty = []
        uttpred = []
        sysy = []
        syspred = []
        ids = []
        if self.metric_type == 'all':
            for id in self.data['utt']['y'].keys():
                utty.append(self.data['utt']['y'][id])
                uttpred.append(self.data['utt']['pred'][id])
                ids.append([id] * len(self.data['utt']['pred'][id]))
            for id in self.data['sys']['y'].keys():
                sysy.append(self.data['sys']['y'][id])
                syspred.append(self.data['sys']['pred'][id])
            utty = np.hstack(utty)
            uttpred = np.hstack(uttpred)
            sysy = list(itertools.chain.from_iterable(sysy))
            syspred = list(itertools.chain.from_iterable(syspred))
            ids = list(itertools.chain.from_iterable(ids))
        elif self.metric_type == 'mean_utt':
            for id in self.data['utt']['y'].keys():
                utty.append(np.mean(self.data['utt']['y'][id]))
                uttpred.append(np.mean(self.data['utt']['pred'][id]))
                ids.append(id)
            for id in self.data['sys']['y'].keys():
                sysy.append(np.mean(self.data['sys']['y'][id]))
                syspred.append(np.mean(self.data['sys']['pred'][id]))
            utty = np.hstack(utty)
            uttpred = np.hstack(uttpred)
            sysy = np.hstack(sysy)
            syspred = np.hstack(syspred)
        else:
            raise NotImplementedError(f'not supported metric_type {self.metric_type}')

        metrics = get_metrics(utty, uttpred, sysy, syspred)
        record = self.metric(*metrics)
        if self.best_record == None or self.best_record < record:
            self.best_metrics = metrics
            self.best_record = record
            self.best_utty = utty
            self.best_uttpred = uttpred
            self.best_id = ids
            self.best_ep = ep
        print('uttlcc\tuttsrcc\tuttmse\tsyslcc\tsyssrcc\tsysmse')
        print('\t'.join(list(map(lambda x: f'{x:.3f}', metrics))))
        self.data = defaultdict(lambda :defaultdict(lambda :defaultdict(list)))

    def write_best(self):
        with open(f'{self.exp}/best.output', 'w+') as wo, open(f'{self.exp}/best.metric', 'w+') as wm:
            for id, y, pred in zip(self.best_id, self.best_utty, self.best_uttpred):
                wo.write(f'{id},{y},{pred}\n')
            wm.write('\t'.join(list(map(str, self.best_metrics))))
