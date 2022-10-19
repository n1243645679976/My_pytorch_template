import os
import yaml
import torch
import librosa
import importlib
import shutil
import random
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from utils.get_rand import get_random
from utils.dynamic_import import dynamic_import
import math
import functools
import itertools

class packed_batch():
    def __init__(self, data, data_len=None):
        self.data = data
        self.len = data_len
    def __repr__(self):
        return f'{self.data=}, {self.len=}'
        return f'data shape: {self.data.shape}, length of data: {self.len}'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature_dir, data, conf, stage, train_data=None, extract_feature_online=False, device='cpu'):
        """
        conf['feature']: [feature_1, feature_2, ..., feature_n] (e.g. ['spectrogram', 'hubert', 'score.txt'])
        conf['label']:   [feature_1, feature_2] (e.g. ['score.txt'])
        
        feature is *.txt will be opened when __init__, the first column of the file will be the key, and the second column of the file will be cast to float
        feature is *.emb, then will convert second column to id(increase from 0 by step 1), if *.embid exist, then read it
        feature is *.list, then will convert columns from second to the end to List[Float]
        else, feature will be lazily loaded

        we open data/{data}/wav.scp as filelist, and read feature from exp/{data}/{feat}/*.pt
        """
        print(f'read dataset {data}')
        self.conf = conf
        self.key_list = [] # for iter in __getitem__
        self.feat_list = conf['features'] # for iter in __getitem__
        self.stage = stage
        self.label_list = conf.get('label', []) # for exclude in __getitem__
        self.extract_feature_online = (extract_feature_online.lower() == 'true')     # for iter in __getitem__

        self.feature_dir = feature_dir
        self.extractors = {}

        self.dir = data
        self.data = {}
        self.taken_item = conf.get('pairing', {'additional_data': 0})['additional_data'] + 1
        
        self.collate_fns = conf['collate_fns'] # for get_dataloader use
        self.batch_size = conf['batch_size'].get(stage, conf['batch_size']['_default']) # for get_dataloader use
        self.augment = defaultdict(list)
        self.feat_key_wav_dict = defaultdict(lambda : defaultdict(lambda : defaultdict(None)))
        self.trial_keys = defaultdict(lambda : defaultdict(None))
        self.wavdict = defaultdict(list)
        self.device = device

        for aug_feature, aug_confs in conf.get('data_augmentation', {}).items():
            for aug_conf in aug_confs:
                aug_class = dynamic_import(aug_conf['augment'])
                aug_method = aug_class(**aug_conf['conf'])
                self.augment[aug_feature].append(aug_method)

        # use 'files_id' or 'wav.scp' to retrieve file key for iteration
        if os.path.isfile(os.path.join('data', data, 'files_id')):
            files_id_list = os.path.join('data', data, 'files_id')
        else:
            files_id_list = os.path.join('data', data, 'wav.scp')
        with open(files_id_list) as f:
            for line in f.read().splitlines():
                 self.key_list.append(line.split()[0])

        for feat in conf['features']:
            ## txt: {id} {float}
            if feat.endswith('.txt'):
                with open(os.path.join('data', data, feat)) as f:
                    self.data[feat] = {}
                    for line in f.read().splitlines():
                        key, value = line.split()
                        self.data[feat][key] = torch.tensor(float(value)).reshape(1)

            ## emb: {id} {some identifier}
            ##   diction write in to .embid
            elif feat.endswith('.emb'):
                id = 0
                key2value = {}
                if train_data:
                    shutil.copy(os.path.join('data', train_data, feat + 'id'), os.path.join('data', data, feat + 'id'))

                if os.path.isfile(os.path.join('data', data, feat + 'id')):
                    with open(os.path.join('data', data, feat + 'id')) as f:
                        for line in sorted(f.read().splitlines()):
                            emb, _id = line.split()
                            _id = int(_id)
                            key2value[emb] = _id
                            id = max(id, _id + 1)

                with open(os.path.join('data', data, feat)) as f:
                    self.data[feat] = {}
                    for line in sorted(f.read().splitlines()):
                        key, emb = line.split()
                        if emb not in key2value:
                            key2value[emb] = id
                            id += 1
                        self.data[feat][key] = torch.tensor(key2value[emb]).long().reshape(1)

                with open(os.path.join('data', data, feat + 'id'), 'w+') as w:
                    for emb, _id in key2value.items():
                        w.write(f'{emb} {_id}\n')
            
            ## list: {id} {float1} {float2} ...
            ## e.g. wav1 1.1 2.1 3.1
            ##     -> [1.1, 2.1, 3.1]
            elif feat.endswith('.list'):
                with open(os.path.join('data', data, feat)) as f:
                    self.data[feat] = {}
                    for line in f.read().splitlines():
                        key, value = line.split(' ', maxsplit=1)
                        self.data[feat][key] = torch.tensor(list(map(float, value.split()))).reshape(-1)

            ## listemb: {id} {token1} {token2} {token3}
            ## e.g. wav1 aa b c
            ##      wav2 aa aa d
            ##   -> wav1: [1, 2, 3]
            ##   -> wav2: [1, 1, 4]
            elif feat.endswith('.listemb'):
                id = 1 # start from 1 to avoid pad by 0
                key2value = {}
                if train_data:
                    shutil.copy(os.path.join('data', train_data, feat + 'id'), os.path.join('data', data, feat + 'id'))

                if os.path.isfile(os.path.join('data', data, feat + 'id')):
                    with open(os.path.join('data', data, feat + 'id')) as f:
                        for line in sorted(f.read().splitlines()):
                            emb, _id = line.split()
                            _id = int(_id)
                            key2value[emb] = _id
                            id = max(id, _id + 1)

                with open(os.path.join('data', data, feat)) as f:
                    self.data[feat] = {}
                    for line in sorted(f.read().splitlines()):
                        key, embs = line.split(maxsplit=1)
                        listemb = []
                        for emb in embs.split():
                            if emb not in key2value:
                                key2value[emb] = id
                                id += 1
                            listemb.append(torch.tensor(key2value[emb]).long().reshape(1))
                        self.data[feat][key] = torch.cat(listemb, dim=0)

                with open(os.path.join('data', data, feat + 'id'), 'w+') as w:
                    for emb, _id in key2value.items():
                        w.write(f'{emb} {_id}\n')
            
            ## listchar: {id} {token1}{token2}{token3}
            ## e.g. wav1 abc
            ##      wav2 ab cd
            ##   generate dict: {'a':0, 'b':1, 'c':2, ' ':3, 'd':4}
            ##   -> wav1: [1, 2, 3]
            ##   -> wav2: [1, 2, 4, 3, 5]
            elif feat.endswith('.listchar'):
                id = 1 # start from 1 to avoid pad by 0
                key2value = {}
                if train_data:
                    shutil.copy(os.path.join('data', train_data, feat + 'id'), os.path.join('data', data, feat + 'id'))

                if os.path.isfile(os.path.join('data', data, feat + 'id')):
                    with open(os.path.join('data', data, feat + 'id')) as f:
                        for line in sorted(f.read().splitlines()):
                            emb, _id = line[0], line[2:]
                            _id = int(_id)
                            key2value[emb] = _id
                            id = max(id, _id + 1)

                with open(os.path.join('data', data, feat)) as f:
                    self.data[feat] = {}
                    for line in sorted(f.read().splitlines()):
                        key, embs = line.split(maxsplit=1)
                        listemb = []
                        for emb in embs:
                            if emb not in key2value:
                                key2value[emb] = id
                                id += 1
                            listemb.append(torch.tensor(key2value[emb]).long().reshape(1))
                        self.data[feat][key] = torch.cat(listemb, dim=0)

                with open(os.path.join('data', data, feat + 'id'), 'w+') as w:
                    for emb, _id in key2value.items():
                        w.write(f'{emb} {_id}\n')
                        

            elif feat.split('#')[0].endswith('random'):
                self.data[feat][key] = get_random(feat)
            ## others:
            ##   e.g.  spectrogram#wav.scp, spectrogram#clean_wav.scp -> use conf/spectrogram.yaml to extract data/{data}/wav.scp or data/{data}/clean_wav.scp
            ##         stoi#trial#wav.scp#wav1.scp, pesq#trial#wav.scp#wav1.scp
            ##   only the shape of dimension 0 can be variable
            else:
                if self.extract_feature_online:
                    feat_conf_file = feat.split('#')[0]
                    with open(f'conf/features/{feat_conf_file}.yaml') as f:
                        feat_conf = yaml.safe_load(f)
                        feat_conf['fs'] = conf['fs']
                    self.extractors[feat] = importlib.import_module(f'model.features.{feat_conf["feature"]}').extractor(feat_conf)

                    # Not given file to extract, then use default: 'wav.scp' or 'trial', 'wav.scp', 'wav1.scp', 'wav2.scp', ...
                    self.filelist = feat.split('#')[1:]
                    if self.filelist == []:
                        self.filelist = self.extractors[feat].get_default_input_filenames()

                    if len(self.filelist) == 1:
                        with open(os.path.join('data', data, self.filelist[0])) as f:
                            for line in f.read().splitlines():
                                key, value = line.split()
                                self.feat_key_wav_dict[feat][key][0] = value
                                self.trial_keys[feat][key] = [key]
                    else:
                        for i, file in enumerate(self.filelist[1:]):
                            with open(os.path.join('data', data, file)) as f:
                                for line in f.read().splitlines():
                                    key, value = line.split()
                                    self.feat_key_wav_dict[feat][key][i] = value

                        with open(os.path.join('data', data, self.filelist[0])) as f:
                            for line in f.read().splitlines():
                                key, value = line.split(' ', maxsplit=1)
                                self.trial_keys[feat][key] = value.split()

                self.data[feat] = defaultdict(lambda : None)
                
        if self.taken_item > 1:
            self.limits = conf['pairing']['limits'].split(',')
            self.choose_list = {}
            for i, key in enumerate(self.key_list):
                node = self.get_limit_node(key)
                if 'data' not in node:
                    node['data'] = []
                node['data'].append(i)

    def __len__(self):
        return len(self.key_list)

    def get_limit_node(self, key):
        node = self.choose_list
        for limit in self.limits:
            # TODO: maybe we can support different latter but it may be difficult
            if limit.startswith('same:'):
                limit = limit[5:]
                data = self.data[limit][key]
                data = tuple(float(d) for d in data) # one dimension
                if data not in node:
                    node[data] = {}
                node = node[data]
            else:
                raise Exception(f'unknow limit: {limit}')
        return node
    
    def __getitem__(self, index):
        ret = [self.getitem(index)]
        if self.taken_item > 1:
            for i in range(self.taken_item - 1):
                node = self.get_limit_node(self.key_list[index])
                chooseind = random.choice(node['data'])
                ret.append(self.getitem(chooseind))
        return ret
    
    def getitem(self, index):
        x, y = [], []
        key = self.key_list[index]
        for feat in self.feat_list:
            if self.stage == 'test':
                if feat in self.label_list:
                    continue
            if feat.split('#')[0].endswith('random'):
                data = self.data[feat][key]()
            else:
                data = self.data[feat][key]
                if data == None:
                    if self.extract_feature_online:
                        keys = self.trial_keys[feat][key]
                        wavs = []
                        for i, _key in enumerate(keys):
                            wavfile = self.feat_key_wav_dict[feat][_key][i]
                            if len(self.wavdict[wavfile]) == 0:
                                # we load wavfile with librosa.load, which can resmaple the wavform in the same time.
                                if wavfile.endswith('.wav') or wavfile.endswith('.flac'):
                                    wav, sr = librosa.load(wavfile, sr=self.conf['fs'])
                                    self.wavdict[wavfile] = wav
                                elif wavfile.endswith('.pt'):
                                    wav = torch.load(wavfile)
                                # you can add some condition like ".endswtih('.png')" or other extensions here to read it by some file reading method
                            wavs.append(self.wavdict[wavfile])
                        self.data[feat][key] = self.extractors[feat](*wavs)
                    else:
                        feat_dir = feat.split('#')[0]
                        self.data[feat][key] = torch.load(os.path.join(self.feature_dir, self.dir, feat_dir, f'{key}.pt')).float()
                    data = self.data[feat][key]
             
            if self.stage == 'train':      
                for aug_method in self.augment[feat]:
                    data = aug_method(data)
                    
            if feat in self.label_list:
                y.append(data)
            else:
                x.append(data)
        return x, y, key

    def get_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True,
                          collate_fn=get_collate_fn(self.collate_fns, self.device, self.feat_list, self.label_list, self.stage),
                         )

def get_collate_fn(conf, device, features, label_list, stage):
    
    def aggregate_pad_to_max(batch, i, xyind):
        x, lenx = [], []
        for data in itertools.chain.from_iterable(batch):
            x.append(data[xyind][i].to(device))
            lenx.append(data[xyind][i].shape[0])
        x = pad_sequence(x, batch_first=True)
        lenx = torch.tensor(lenx)
        return packed_batch(x, lenx)
    
    def aggregate_repetitive_to_max(batch, i, xyind):
        x, lenx = [], []
        maxlen = max(data[xyind][i].shape[0] for data in itertools.chain.from_iterable(batch))
        for data in itertools.chain.from_iterable(batch):
            span = [math.ceil(maxlen/data[xyind][i].shape[0])] + [1] * (data[xyind][i].dim()-1)
            padding_tensor = data[xyind][i].repeat(*span)[:maxlen]
            x.append(padding_tensor.to(device).unsqueeze(0))
            lenx.append(maxlen)
        x = torch.cat(x, dim=0)
        lenx = torch.tensor(lenx)
        return packed_batch(x, lenx)
        
    def aggregate_crop_to_min_rand(batch, i, xyind):
        x, lenx = [], []
        minlen = min(data[xyind][i].shape[0] for data in itertools.chain.from_iterable(batch))
        for data in itertools.chain.from_iterable(batch):
            start_index = (torch.rand(1) * (data[xyind][i].shape[0] - minlen + 1)).long()
            x.append(data[xyind][i][start_index:start_index+minlen].to(device).unsqueeze(0))
            lenx.append(minlen)
        x = torch.cat(x, dim=0)
        lenx = torch.tensor(lenx)
        return packed_batch(x, lenx)
    
    def aggregate_crop_to_min(batch, i, xyind):
        x, lenx = [], []
        minlen = min(data[xyind][i].shape[0] for data in itertools.chain.from_iterable(batch))
        for data in itertools.chain.from_iterable(batch):
            start_index = 0
            x.append(data[xyind][i][start_index:start_index+minlen].to(device).unsqueeze(0))
            lenx.append(minlen)
        x = torch.cat(x, dim=0)
        lenx = torch.tensor(lenx)
        return packed_batch(x, lenx)
    
    xi, yi = 0, 0
    fns = []
    keys = []
    for feature in features:
        aggregate_method = conf.get(feature, conf['_default'])
        if aggregate_method == 'pad_to_max':
            fn = aggregate_pad_to_max
        elif aggregate_method == 'repetitive_to_max':
            fn = aggregate_repetitive_to_max
        elif aggregate_method == 'crop_to_min':
            fn = aggregate_crop_to_min
        elif aggregate_method == 'crop_to_min_rand':
            fn = aggregate_crop_to_min_rand
        else:
            raise Exception(f'not supported aggregate_method: {aggregate_method}')
        if feature in label_list:
            if stage == 'test':
                continue
            fns.append(functools.partial(fn, i=yi, xyind=1))
            keys.append(f'_dataset_feat_y{yi}')
            yi += 1
        else:
            fns.append(functools.partial(fn, i=xi, xyind=0))
            keys.append(f'_dataset_feat_x{xi}')
            xi += 1
    
    def collate_fn(batch):
        data = {}
        ids = []
        for key, fn in zip(keys, fns):
            data[key] = fn(batch)
        for _x, _y, _id in itertools.chain.from_iterable(batch):
            ids.append(_id)
        data['_ids'] = ids
        return data
    
    return collate_fn

if __name__ == '__main__':
    #from utils.parse_config import get_train_config
    #args, conf = get_train_config()
    #dataloader = Dataset(args.features, data=args.train, conf=conf['dataset'], extract_feature_online=True).get_dataloader()
    #for x in dataloader:
    #    print(x)
    for method in ['pad_to_max', 'repetitive_to_max', 'crop_to_min', 'crop_to_min_rand']:
        print(f'\n\n\n\n{method=}')
        collate_fn = get_collate_fn({'_default': method, '3': 'pad_to_max'}, 'cpu', ['1', '2', '3'], ['3'], 'train')
        batch = [ [ [ torch.arange(20000).reshape(100, 200), torch.randn(1)], [torch.arange(13).float()], 'temp1' ], \
                [ [ torch.arange(18000).reshape(90, 200), torch.randn(1)], [torch.arange(17)], 'temp2' ], \
                [ [ torch.arange(16000).reshape(80, 200), torch.randn(1)], [torch.arange(17)], 'temp3' ] ]
        print(collate_fn(batch))
    

