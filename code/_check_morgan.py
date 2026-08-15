# -*- coding: utf-8 -*-
"""检查 chem_morgan.pkl 是否覆盖 test 新化合物"""
import numpy as np, pandas as pd, pickle

DATA = 'data'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
tmeta = pd.read_csv('../input/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')

# chem_morgan.pkl 结构检查
with open(f'{DATA}/chem_morgan.pkl', 'rb') as f:
    cm = pickle.load(f)
print('chem_morgan.pkl type:', type(cm))
if isinstance(cm, dict):
    keys = list(cm.keys())[:5]
    print('keys 样例:', keys)
    print('key 数:', len(cm))
    # 检查 test 化合物是否在其中
    t_chems = sorted(tmeta['perturbation_no_concentration'].unique())
    hit = sum(1 for c in t_chems if c in cm)
    print(f'test 化合物命中 chem_morgan.pkl: {hit}/{len(t_chems)}')
    for c in t_chems:
        if c in cm:
            print(f'  {c}: OK')
elif isinstance(cm, np.ndarray):
    print('shape:', cm.shape)
    # 可能按行索引
    print('checking first rows...')
