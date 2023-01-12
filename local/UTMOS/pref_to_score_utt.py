import sys
from collections import defaultdict, Counter
import numpy as np
preffile = sys.argv[1]
outfile = sys.argv[2]

scores = defaultdict(list)
scorecounter = Counter()
with open(preffile) as f:
    for line in f.readlines():
        uttpair, score = line.split()
        utt1, utt2 = uttpair.split('__&__')
        score = int(score)
        scores[utt1].append(score-1)
        scores[utt2].append(1-score)
        scorecounter[utt1.split('-')[0]] += 1
        scorecounter[utt2.split('-')[0]] += 1


with open(outfile, 'w+') as f:
    for key in scores:
        f.write(f'{key},{np.mean(scores[key])}\n')

