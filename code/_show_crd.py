# -*- coding: utf-8 -*-
import gzip, numpy as np
raw = gzip.open('data/1011_SNP_distance.tab.gz','rb').read().decode('utf-8')
lines = raw.strip().split('\n')
header = lines[0].strip().split('\t')
cols = header
strains5 = ['BAH','BAI','CEK','CGD','CRD']
idx = [cols.index(s) for s in strains5]
D = {}
for line in lines[1:]:
    parts = line.strip().split('\t')
    rname = parts[0]
    if rname in strains5:
        D[rname] = {strains5[j]: float(parts[idx[j]]) for j in range(5)}

print('遗传距离矩阵（SNP 距离，越小越近）:')
header_line = '        ' + ''.join(f'{s:>10}' for s in strains5)
print(header_line)
for s in strains5:
    row = ''.join(f'{D[s][t]:>10.4f}' for t in strains5)
    print(f'{s:>8}' + row)

print()
print('CRD 与训练菌株的距离（升序）:')
crd = {t: D['CRD'][t] for t in ['BAH','BAI','CEK','CGD']}
for t, d in sorted(crd.items(), key=lambda x: x[1]):
    print(f'  {t}: {d:.4f}')

print()
train_pairs = [D[a][b] for a in ['BAH','BAI','CEK','CGD'] for b in ['BAH','BAI','CEK','CGD'] if a < b]
print('训练菌株之间平均距离:', round(np.mean(train_pairs), 4))
print('CRD 到训练菌株平均距离:', round(np.mean(list(crd.values())), 4))
print('最近训练菌株:', min(crd, key=crd.get), f'({crd[min(crd, key=crd.get)]:.4f})')
