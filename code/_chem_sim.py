# -*- coding: utf-8 -*-
"""检查 test 新化合物与训练化合物的 Morgan 相似度"""
import numpy as np, pandas as pd, pickle

DATA = 'data'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))
morgan64 = feats['chem_morgan']
tmeta = pd.read_csv('../input/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')

pert2morgan64 = {}
for i, p in enumerate(meta['perturbation_no_concentration'].values):
    pert2morgan64.setdefault(p, morgan64[i])

idx = np.where(tmeta['split_final'].eq('test_chem_only').values)[0]
t_chems = sorted(tmeta.iloc[idx]['perturbation_no_concentration'].unique())
train_chems = sorted(set(meta.loc[meta['role'].eq('treatment'), 'perturbation_no_concentration']))

for c in t_chems:
    if c not in pert2morgan64:
        print(f'{c}: 无指纹')
        continue
    v = pert2morgan64[c]
    sims = []
    for tc in train_chems:
        if tc not in pert2morgan64:
            continue
        w = pert2morgan64[tc]
        s = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-8)
        sims.append(s)
    sims = np.array(sims)
    top = np.argsort(-sims)[:3]
    print(f'{c}: max_sim={sims.max():.3f} top3={[train_chems[i] for i in top]} sims={np.round(sims[top], 2)}')
