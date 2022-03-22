import os
import yaml
import torch
import librosa
import importlib
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class packed_batch():
    def __init__(self, data, data_len=None):
        self.data = data
        self.len = data_len
    def __repr__(self):
        return f'data shape: {self.data.shape}, length of data: {self.len}'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature_dir, data, conf, extract_feature_online=False, device='cpu', inferring=False):
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
        self.inferring = inferring # for iter in __getitem__
        self.label_list = conf.get('label', []) # for exclude in __getitem__
        self.extract_feature_online = extract_feature_online # for iter in __getitem__

        self.feature_dir = feature_dir
        self.extractors = {}

        self.dir = data
        self.data = {}
        self.collate_fn = conf['collate_fn'] # for get_dataloader use
        self.batch_size = conf['batch_size'] # for get_dataloader use
        self.feat_key_wav_dict = defaultdict(lambda : defaultdict(lambda : defaultdict(None)))
        self.trial_keys = defaultdict(lambda : defaultdict(None))
        self.wavdict = defaultdict(list)
        self.device = device

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
            
            ## list: {id} {float1} {float2} ...
            ## e.g. wav1 1.1 2.1 3.1
            ##     -> [1.1, 2.1, 3.1]
            elif feat.endswith('.list'):
                with open(os.path.join('data', data, feat)) as f:
                    self.data[feat] = {}
                    for line in f.read().splitlines():
                        key, value = line.split(' ', maxsplit=1)
                        self.data[feat][key] = torch.tensor(list(map(float, value.split()))).reshape(-1)
            
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
                
    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, index):
        x = []
        y = []
        key = self.key_list[index]
        for feat in self.feat_list:
            if self.inferring:
                if feat in self.label_list:
                    continue
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
                            # you can add some condition lik e".endswtih('.png')" or other here to read it by some file reading method
                        wavs.append(self.wavdict[wavfile])
                    self.data[feat][key] = self.extractors[feat](*wavs)
                else:
                    self.data[feat][key] = torch.load(os.path.join(self.feature_dir, self.dir, feat, f'{key}.pt')).float()
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
        batched_data = {}

        ids = []
        for i in range(len(batch[0][0])):
            x, lenx = [], []
            for _x, _y, _id in batch:
                x.append(_x[i].to(device))
                lenx.append(_x[i].shape[0])
            x = pad_sequence(x, batch_first=True)
            lenx = torch.tensor(lenx)
            batched_data[f'_dataset_feat_x{i}'] = packed_batch(x, lenx)

        for i in range(len(batch[0][1])):
            y, leny = [], []
            for _x, _y, _id in batch:
                y.append(_y[i].to(device))
                leny.append(_y[i].shape[0])
            y = pad_sequence(y, batch_first=True)
            leny = torch.tensor(leny)
            batched_data[f'_dataset_feat_y{i}'] = packed_batch(y, leny)

        for _x, _y, _id in batch:
            ids.append(_id)
        batched_data[f'_ids'] = ids
        return batched_data
    return collate_fn

if __name__ == '__main__':
    from utils.parse_config import get_train_config
    args, conf = get_train_config()
    dataloader = Dataset(args.features, data=args.train, conf=conf['dataset'], extract_feature_online=True).get_dataloader()
    for x in dataloader:
        print(x)