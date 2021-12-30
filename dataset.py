import os
import torch
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, exp, data, conf, device='cpu', inferring=False):
        """
        conf['feature']: [feature_1, feature_2, ..., feature_n] (e.g. ['spectrogram', 'hubert', 'score.txt'])
        conf['label']:   [feature_1, feature_2] (e.g. ['score.txt'])
        
        feature is *.txt will be opened when __init__, the first column of the file will be the key, and the second column of the file will be cast to float
        feature is *.emb, then will convert second column to id(increase from 0 by step 1), if *.embid exist, then read it
        feature is *.list, then will convert columns from second to the end to List[Float]
        else, feature will be lazily loaded

        we open data/{data}/wav.scp as filelist, and read feature from exp/{data}/{feat}/*.pt
        """
        self.exp = exp
        self.dir = data
        self.data = {}
        self.feat_list = conf['feature']
        self.label_list = conf['label']
        self.collate_fn = conf['collate_fn']
        self.exclude_inference = conf['exclude_inference']
        self.batch_size = conf['batch_size']
        self.inferring = inferring
        self.file_list = []
        self.device = device
        with open(os.path.join('data', data, 'wav.scp')) as f:
            for line in f.read().splitlines():
                 self.file_list.append(line.split()[0])

        for feat in conf['feature']:
            if feat.endswith('.txt'):
                with open(os.path.join('data', data, feat)) as f:
                    self.data[feat] = {}
                    for line in f.read().splitlines():
                        key, value = line.split()
                        self.data[feat][key] = torch.tensor(float(value)).reshape(1)

            elif feat.endswith('.emb'):
                id = 0
                if os.path.isfile(os.path.join('data', data, feat + 'id')):
                    with open(os.path.join('data', data, feat + 'id')) as f:
                        for line in sorted(f.read().splitlines()):
                            key, _id = line.split()
                            self.data[feat][key] = torch.tensor(_id).long().reshape(1)
                        id = _id + 1
                with open(os.path.join('data', data, feat)) as f:
                    self.data[feat] = {}
                    for line in sorted(f.read().splitlines()):
                        key, value = line.split()
                        if key not in self.data[feat]:
                            self.data[feat][key] = torch.tensor(id).long().reshape(1)
                            id += 1
                with open(os.path.join('data', data, feat + 'id'), 'w+') as w:
                    for key, value in self.data[feat].items():
                        w.write(f'{key} {value}\n')
                
            elif feat.endswith('.list'):
                with open(os.path.join('exp', data, feat)) as w:
                    self.data[feat] = {}
                    for line in f.read().splitlines():
                        key = line.split()[0]
                        value = line.split()[1:]
                        self.data[feat][key] = list(map(lambda x: torch.tensor(float(x)).reshape(1), value))
            else:
                self.data[feat] = defaultdict(lambda : None)
        print(self.data)
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        x = []
        y = []
        key = self.file_list[index]
        for feat in self.feat_list:
            if self.inferring:
                if feat in self.exclude_inference:
                    continue
            data = self.data[feat][key]
            if data == None:
                self.data[feat][key] = torch.load(os.path.join(self.exp, self.dir, feat, f'{key}.pt')).float()
                data = self.data[feat][key]
            if feat in self.label_list:
                y.append(data)
            else:
                x.append(data)
        return x, y, key

    def get_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True,
                          collate_fn=get_collate_fn(self.collate_fn, self.device),
                         )

def get_collate_fn(conf, device):
    def collate_fn(batch):
        """
        batch: [(x1, y1, id1), (x2, y2, id2), ...], which equals to [((feat1_x1, feat2_x1,...), (feat1_y1, feat2_y1,...), 'id1'), ((feat1_x2, feat2_x2,...), (feat1_y2, feat2_y2,...), 'id2'), ...]
        output: [[[feat1_x1, feat1_x2,...], [feat2_x1, feat2_x2,...],...], [[feat1_y1, feat1_y1,...], [feat2_y1, feat2_y2,...],...], ['id1', 'id2',...]]
        """
        x = []
        y = []
        id = []
        for i in range(len(batch[0][0])):
            x.append([])
            y.append([])
            for _x, _y, _id in batch:
                x[-1].append(_x[i].to(device))
                y[-1].append(_y[i].to(device))
                id.append(id)
            if conf == 'pad_to_max':
                x[-1] = pad_sequence(x[-1], batch_first=True)
                y[-1] = pad_sequence(y[-1], batch_first=True)
        return x, y, id
    return collate_fn
        
if __name__ == '__main__':
    from utils.parse_config import get_config
    args, conf = get_config()
    dataloader = Dataset(args.exp, data=args.train, conf=conf['dataset']).get_dataloader()
    for x, y, id in dataloader:
        print(x, y, id)
