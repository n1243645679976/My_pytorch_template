import sys
import random
import itertools 
from collections import defaultdict, Counter, deque
set_name = sys.argv[1]
total_pairs = int(sys.argv[2])

sys2utt = defaultdict(list)
sysset = set()
ind = 0
with open(f'data/{set_name}/wav.scp') as f:
    for line in f.read().splitlines():
        key, v = line.split()
        sys = key.split('-')[0]
        sys2utt[sys].append(key)
        sysset.add(sys)

    ind2sys = list(sysset)
    sys2ind = {b:a for a, b in enumerate(ind2sys)}

    sys_pair_count = defaultdict(lambda:defaultdict(int))
    for sys1 in sysset:
        for sys2 in sysset:
            sys_pair_count[sys1][sys2] = len(sys2utt[sys1]) * len(sys2utt[sys2])

    with open(f'data/{set_name}/trials', 'w+') as w:
        syscomb = list(itertools.combinations(sysset, 2))
        syss = [tuple(sorted([sys1, sys2])) for sys1, sys2 in random.choices(syscomb, k=total_pairs)]
        syssCounter = Counter(syss)
        ok = []
        notok = []
        for (sys1, sys2), ps in syssCounter.items():
            utt_pairs = list(itertools.product(sys2utt[sys1], sys2utt[sys2]))
            for utt1, utt2 in random.sample(utt_pairs, k=min(ps, len(utt_pairs))):
                ok.append((utt1, utt2, sys1, sys2))
                sys_pair_count[sys1][sys2] -= 1
                sys_pair_count[sys2][sys1] -= 1
            for i in range(len(utt_pairs), ps):
                notok.append((utt1, utt2, sys1, sys2))

        for utt1, utt2, sys1, sys2 in notok:
            r = random.choice(range(len(ok)))
            count = 0
            utt3, utt4, sys3, sys4 = ok[r]
            while count < 10000:
                if len(set([sys1, sys2, sys3, sys4])) == 4:
                    p1, p2 = sorted([sys1, utt1, sys3, utt3]), sorted([sys2, sys4, utt2, utt4])
                    if sys_pair_count[p1[0]][p1[2]] > 0 and sys_pair_count[p2[0]][p2[2]] > 0:
                        break
                    p1, p2 = sorted([sys1, sys4, utt1, utt4]), sorted([sys2, sys3, utt2, utt3])
                    if sys_pair_count[p1[0]][p1[2]] > 0 and sys_pair_count[p2[0]][p2[2]] > 0:
                        break
                r -= 1
                utt3, utt4, sys3, sys4 = ok[r]
                count += 1
            if count == 10000:
                raise Exception('retry 10001 times')
            ok.pop(r)
            ok.append((p1[1], p1[3], '', '' ))
            ok.append((p2[1], p2[3], '', ''))
            sys_pair_count[p1[0]][p1[2]] -= 1
            sys_pair_count[p2[0]][p2[2]] -= 1
            sys_pair_count[sys3][sys4] += 1
        for utt1, utt2, _, _ in ok:
            w.write(f'{utt1} {utt2}\n')





            
