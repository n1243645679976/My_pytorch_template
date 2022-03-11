import sys
best_iter = -1
best_met = float('-inf')
out = ''
def get_met(met):
    return met[3]+met[4]+met[6]+met[7]
with open(sys.argv[1]) as f:
    for line in f.readlines()[1:]:
        metrics = list(map(float, line.split()))
        if get_met(metrics) > best_met:
            best_met = get_met(metrics)
            best_iter = metrics[0]
            out = line
print(best_met, best_iter)
print(out)


