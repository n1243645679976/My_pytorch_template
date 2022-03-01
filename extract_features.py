from collections import defaultdict
import os
import sys
import tqdm
import torch
import numpy as np
from utils.parse_config import get_feat_config
from dataset import Dataset

if __name__ == "__main__":
    args, conf = get_feat_config()
    assert 'label' not in conf['dataset'] or conf['dataset']['label'] == [], 'conf["dataset"]["label"] is not empty. Please remove it to force all features go to batchxs'
    # feature extraction is done in dataset with extract_feature_online
    dataloader = Dataset(feature_dir=args.outdir, data=args.set, conf=conf['dataset'], extract_feature_online=True, device=args.device).get_dataloader()
    save_formats = [(f, 'unique') if len(f.split('#')) == 1 else f.split('#', maxsplit=1) for f in conf['save_format']]
    values = defaultdict(lambda :defaultdict(list))

    for batchxs, _, keys in tqdm.tqdm(dataloader):
        # iterate keys, we want to save all features with its corresponding keys
        for features, feature_name, (save_format, aggregation_type) in zip(batchxs, conf['dataset']['features'], save_formats):
            # maybe we don't extract only one feature a time.
            # iterate feature and its corresponding feature_name and save_format
            for feature, key in zip(features.data, keys):
                if aggregation_type == 'unique':
                    assert key not in values[feature_name], "key doesn't exist uniquely"
                if save_format == 'pt':
                    values[feature_name][key].append(feature.unsqueeze(0))
                else:
                    print(feature_name, key)
                    values[feature_name][key].append(feature.detach().cpu().numpy())
    
    # we only support 'pt' and 'txt' save format
    for feature_name, (save_format, aggregation_type) in zip(conf['dataset']['features'], save_formats):
        save_feature_name = feature_name.split('#')[0]
        if save_format == 'pt':
            os.makedirs(os.path.join(args.outdir, args.set, save_feature_name), exist_ok=True)
            for key in values[feature_name].keys():
                if aggregation_type == 'mean':
                    feature = torch.mean(values[feature_name][key])
                torch.save(feature, os.path.join(args.outdir, args.set, save_feature_name, key + '.pt'), _use_new_zipfile_serialization=False)
        elif save_format == 'txt':
            with open(os.path.join('data', args.set, save_feature_name + '.txt'), 'w+') as w:
                for key in values[feature_name].keys():
                    # TODO: since the dataloader can only output one value per key, this is redundant if we don't modify dataloader to support multi-output per key.
                    if aggregation_type == 'mean':
                        feature = np.mean(np.array(values[feature_name][key]))
                    elif aggregation_type == 'max':
                        feature = np.max(np.array(values[feature_name][key]))
                    elif aggregation_type == 'min':
                        feature = np.min(np.array(values[feature_name][key]))
                    elif aggregation_type == 'all':
                        feature = ' '.join(map(lambda x: ' '.join(map(str, x.flatten())), values[feature_name][key]))
                        print(values[feature_name][key])
                    else:
                        raise NotImplementedError(f'Not implemented {aggregation_type=}')
                    w.write(f'{key} {feature}\n')
        else:
            raise NotImplementedError(f'Not implemented save format: {save_format}')
                
    with open(os.path.join(args.outdir, args.set, 'command.txt'), 'w+') as w:
        w.write(' '.join(sys.argv))

