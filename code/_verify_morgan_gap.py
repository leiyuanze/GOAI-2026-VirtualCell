# -*- coding: utf-8 -*-
"""验证：test 新化合物的 Morgan 指纹在 feats.pkl 中是否为零向量"""
import numpy as np, pandas as pd, pickle

DATA = 'data'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))
tmeta = pd.read_csv('../input/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')

morgan64 = feats['chem_morgan']  # (8958, 64) PCA 后
print('feats chem_morgan shape:', morgan64.shape)

# feats 中 chem_morgan 是按 meta 行索引的
pert2morgan = {}
for i, p in enumerate(meta['perturbation_no_concentration'].values):
    pert2morgan.setdefault(p, morgan64[i])

# test 新化合物
idx = np.where(tmeta['split_final'].eq('test_chem_only').values)[0]
t_chems = sorted(tmeta.iloc[idx]['perturbation_no_concentration'].unique())
print(f'\ntest_chem_only 新化合物 {len(t_chems)} 个:')
for c in t_chems:
    v = pert2morgan.get(c)
    if v is None:
        status = '映射缺失(零向量)'
    elif np.abs(v).sum() < 1e-6:
        status = '零向量!'
    else:
        status = f'有值 norm={np.linalg.norm(v):.3f}'
    print(f'  {c}: {status}')

# 用 chem_morgan.pkl 原始 2048 指纹检查
with open(f'{DATA}/chem_morgan.pkl', 'rb') as f:
    cm = pickle.load(f)
print('\nchem_morgan.pkl 原始指纹:')
for c in t_chems:
    v = cm.get(c)
    if v is None:
        print(f'  {c}: pkl 中也无')
    else:
        print(f'  {c}: pkl 有值 norm={np.linalg.norm(v):.3f} shape={np.asarray(v).shape}')
