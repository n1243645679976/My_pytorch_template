import sys
import numpy as np
from collections import defaultdict
import scipy.stats
import torch
from BTL_model import get_BTL_Tree_result
test_set_name = sys.argv[1]
pref_file = sys.argv[2]
# sctype == 1: contain draw and 0 <= score <= 2
# sctype == 2: contain draw and -1 <= score <= 1
sctype = (sys.argv[3].lower() == 'true') + (sys.argv[3].lower() == '_1to-1') * 2
try:
    bound1 = float(sys.argv[4])
    bound2 = float(sys.argv[5])
except Exception as e:
    bound1 = None
    bound2 = None

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
    if '-' in id:
        sys0 = id.split('-')[0]
        sys1 = id.split('__&__')[1].split('-')[0]
    else:
        key0, key1 = id.split('__&__')
        sys0 = key0.split('_')[0] + key0.split('_')[-1]
        sys1 = key1.split('_')[0] + key1.split('_')[-1]
#    print(sys0, sys1)
    
    
    if sctype == 1:
        # bound1 == None -> use original score
        # bound1 != None -> for equal range and eer range
        if bound1 != None:
            sc = (sc > bound1) + (sc > bound2)
        pref_score[sys0].append(sc - 1)
        pref_score[sys1].append(1 - sc)
    elif sctype == 2:
        if bound1 != None:
            sc = (sc > bound1) + (sc > bound2)
        pref_score[sys0].append(sc)
        pref_score[sys1].append(-sc)
    else:
        # no draw, use 0
        pref_score[sys0].append(sc)
        pref_score[sys1].append(1-sc)
            
#    sc = (sc > -0.33) + (sc > 0.33)
#    sc = (sc > -0.15663141) + (sc > 0.16249835)
#    sc = (sc > -0.15657955) + (sc > 0.15238206)
#    sc = sc > 0
    sc += 1
    pref_score[sys0].append(sc - 1)
    pref_score[sys1].append(1 - sc)
    
with open(f'data/{test_set_name}/score_norm.txt') as f:
    for line in f.readlines():
        utt, score = line.split()
        score = float(score)
        if '-' in utt:
            sys = utt.split('-')[0]
        else:
            sys = utt.split('_')[0] + utt.split('_')[-1]
        gt_score[sys].append(score)

with open(pref_file) as f:
    for line in f.read().splitlines():
        id, sc = line.split()
        sc = float(sc)
#        setwinning_rate(id, sc)
        count_score1(id, sc)
#getsys_score()

print(get_srcc(gt_score, pref_score), get_lcc(gt_score, pref_score))
#for sys in gt_score:
#    print(sys, np.mean(gt_score[sys]), np.mean(pref_score[sys]))

