import argparse
from collections import defaultdict
import numpy as np
import glob
parser = argparse.ArgumentParser()
parser.add_argument('data', help='data to append mean')
args = parser.parse_args()

domain_utt2score = defaultdict(lambda : defaultdict(list))
with open(f'data/{args.data}/score_norm.txt') as f, open(f'data/{args.data}/domain.emb') as f1:
    for score, domain in zip(f.read().splitlines(), f1.read().splitlines()):
        key1, score = score.split()
        key2, domain = domain.split()
        assert key1 == key2
        key = key1.split('_')[0]
        score = float(score)
        domain_utt2score[domain][key].append(score)

lastkey2value = {}
for file in glob.glob(f'data/{args.data}/*'):
    if file.endswith('id'):
        continue
    with open(file) as f:
        for line in f.read().splitlines():
            key, value = line.split(maxsplit=1)
            key = key.split('_')[0]
            lastkey2value[key] = value
            
    with open(file, 'a+') as w:
        for domain in domain_utt2score:
            for key in domain_utt2score[domain]:
                if file.endswith('judge_id.emb'):
                    w.write(f'{key}_M MEAN_{domain}\n')
                elif file.endswith('score_norm.txt'):
                    w.write(f'{key}_M {np.mean(domain_utt2score[domain][key])}\n')
                else:
                    w.write(f'{key}_M {lastkey2value[key]}\n')




