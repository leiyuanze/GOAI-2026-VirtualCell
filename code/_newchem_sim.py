# -*- coding: utf-8 -*-
"""检查 test 新化合物指纹与训练化合物的相似度分布"""
import numpy as np, pandas as pd, pickle

DATA = 'data'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
tmeta = pd.read_csv('../input/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))
pert2morgan64 = feats['pert2morgan64']

train_chems = sorted(set(meta.loc[meta['role'].eq('treatment'), 'perturbation_no_concentration']))
idx = np.where(tmeta['split_final'].eq('test_chem_only').values)[0]
t_chems = sorted(tmeta.iloc[idx]['perturbation_no_concentration'].unique())

print('test 新化合物 vs 训练化合物 64维指纹 cos 相似度:')
for c in t_chems:
    v = pert2morgan64.get(c)
    if v is None:
        print(f'  {c}: 无指纹')
        continue
    sims = []
    for tc in train_chems:
        w = pert2morgan64.get(tc)
        if w is None:
            continue
        s = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-8)
        sims.append((s, tc))
    sims.sort(reverse=True)
    top3 = [(round(s, 3), tc) for s, tc in sims[:3]]
    print(f'  {c}: max={sims[0][0]:.3f} top3={top3}')
