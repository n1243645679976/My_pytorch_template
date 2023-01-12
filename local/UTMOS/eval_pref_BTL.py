import sys
import numpy as np
from collections import defaultdict
import scipy.stats
import torch
from BTL_model import get_BTL_Tree_result
test_set_name = sys.argv[1]
pref_file = sys.argv[2]

gt_score = defaultdict(list)
pref_score = defaultdict(list)
def get_srcc(d1, d2):
    a1 = []
    a2 = []
    for key in d1.keys():
        a1.append(np.mean(d1[key]))
        a2.append(np.mean(d2[key]))
    return scipy.stats.spearmanr(a1, a2)[0]

def get_lcc(d1, d2):
    a1 = []
    a2 = []
    for key in d1:
        a1.append(np.mean(d1[key]))
        a2.append(np.mean(d2[key]))
    return scipy.stats.pearsonr(a1, a2)[0]

winning_rate = torch.zeros(200,200)
sys2id = {}
id2sys = {}
nowid = 0
def setwinning_rate(id, sc):
    global nowid
    sys0 = id.split('-')[0]
    sys1 = id.split('__&__')[1].split('-')[0]
    if sys0 not in sys2id:
        sys2id[sys0] = nowid; id2sys[nowid] = sys0; nowid += 1
    if sys1 not in sys2id:
        sys2id[sys1] = nowid; id2sys[nowid] = sys1; nowid += 1
#    if sc == 2:
#        winning_rate[sys2id[sys0]][sys2id[sys1]] += 1
    if sc == 0:
        winning_rate[sys2id[sys1]][sys2id[sys0]] += 1
def getsys_score():
    global nowid
    scores = get_BTL_Tree_result(winning_rate, num_players=nowid)
    for i, sc in enumerate(scores):
        pref_score[id2sys[i]] = sc.numpy()
    
def count_score1(id, sc):
    # count by winning - (losing + fair)
    sys0 = id.split('-')[0]
    sys1 = id.split('__&__')[1].split('-')[0]
    pref_score[sys0].append(sc - 1)
    pref_score[sys1].append(1 - sc)
    
with open(f'data/{test_set_name}/score_norm.txt') as f:
    for line in f.readlines():
        utt, score = line.split()
        score = float(score)
        sys = utt.split('-')[0]
        gt_score[sys].append(score)

with open(pref_file) as f:
    for line in f.read().splitlines():
        id, sc = line.split()
        sc = float(sc)
        setwinning_rate(id, sc)
#        count_score1(id, sc)
getsys_score()

print(get_srcc(gt_score, pref_score))
print(get_lcc(gt_score, pref_score))
#for sys in gt_score:
#    print(sys, np.mean(gt_score[sys]), np.mean(pref_score[sys]))

