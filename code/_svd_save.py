# -*- coding: utf-8 -*-
"""gpt2 步骤5：生成 response_mean.npy / explained_variance.npy（train-only SVD）"""
import numpy as np, pandas as pd
from sklearn.decomposition import TruncatedSVD

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy").astype(np.float64)
mask = np.load(f"{DATA}/mask.npy").astype(bool)
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values
tr_y_nan = np.where(mask, y_log2, np.nan)

ctrl_idx = np.where(meta['role'].eq('control').values & train_mask)[0]
ck = (meta.iloc[ctrl_idx]['data_source'].astype(str) + '|' + meta.iloc[ctrl_idx]['instrument'].astype(str) + '|'
      + meta.iloc[ctrl_idx]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_idx]['Strains'].astype(str) + '|'
      + meta.iloc[ctrl_idx]['Medium'].astype(str) + '|' + meta.iloc[ctrl_idx]['Temperature'].astype(str) + '|'
      + meta.iloc[ctrl_idx]['pert_time'].astype(str)).values
lookup = {}
for k, pos in zip(ck, ctrl_idx):
    lookup.setdefault(k, []).append(pos)
treat_all = np.where(meta['role'].eq('treatment').values)[0]

def mc(sid):
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    if k not in lookup:
        return None
    rows = lookup[k]
    cvals = tr_y_nan[rows]; cm = mask[rows] > 0
    return np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)

ctrl = np.full((len(treat_all), P), np.nan)
for i, sid in enumerate(treat_all):
    c = mc(sid)
    if c is not None:
        ctrl[i] = c
pos_of = {sid: i for i, sid in enumerate(treat_all)}
train_treat = np.where(train_mask & meta['role'].eq('treatment').values)[0]
treat_pos = np.array([pos_of[s] for s in train_treat])
delta = np.full((len(treat_pos), P), np.nan)
for i, tp in enumerate(treat_pos):
    if np.isfinite(ctrl[tp]).any():
        delta[i] = tr_y_nan[train_treat[i]] - ctrl[tp]
filled = np.where(np.isfinite(delta), delta, 0.0).astype(np.float32)
svd = TruncatedSVD(n_components=64, random_state=42).fit(filled)
np.save(f"{DATA}/v50_response_mean.npy", np.nanmean(delta, axis=0).astype(np.float32))
np.save(f"{DATA}/v50_explained_variance.npy", svd.explained_variance_ratio_.astype(np.float32))
print("response_mean:", np.load(f"{DATA}/v50_response_mean.npy").shape)
print("explained_variance:", np.load(f"{DATA}/v50_explained_variance.npy").shape, "| 累计方差:", svd.explained_variance_ratio_.sum() * 100, "%")
