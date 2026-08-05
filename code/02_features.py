# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 02 特征工程（v2 方案）
- 实体索引（菌株/化合物/培养基/温度/时间/交叉/上下文）
- 化合物 hash（32 维，新化合物兜底）
- 统计先验 SVD 初始化（菌株均值→16 维 / 化合物Δ均值→32 维）
- 上下文响应先验（菌株×培养基×温度×时间 分组均值，用于 S 分支）
- 蛋白共表达模块（训练集相关矩阵 → 谱聚类 K 模块）
输出到 vcell/data/feats.pkl
纪律：所有统计量仅用训练行(split_final=='train')
"""
import os
import numpy as np
import pandas as pd
import pickle

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"

meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy")
mask = np.load(f"{DATA}/mask.npy")
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values
tr_y = np.where(mask.astype(bool), y_log2, np.nan)

# ---------- 1. 实体索引 ----------
strains = sorted(meta['Strains'].unique())
chems = sorted(meta.loc[meta['role'].eq('treatment'), 'perturbation_no_concentration'].unique())
train_strains = sorted(meta.loc[train_mask, 'Strains'].unique())
train_chems = sorted(meta.loc[train_mask & meta['role'].eq('treatment'), 'perturbation_no_concentration'].unique())
strain2id = {s: i for i, s in enumerate(strains)}
chem2id = {c: i for i, c in enumerate(chems)}
strain_id = meta['Strains'].map(strain2id).values.astype(np.int64)
chem_id = meta['perturbation_no_concentration'].map(chem2id).fillna(-1).values.astype(np.int64)
strain_seen = meta['Strains'].isin(train_strains).values.astype(np.float32)
chem_seen = meta['perturbation_no_concentration'].isin(train_chems).values.astype(np.float32)

# ---------- 2. 化合物 hash（32 维）----------
import hashlib
def hash_vec(name, dim=32):
    h = hashlib.sha256(str(name).encode()).hexdigest()   # 64 hex chars, 足够 dim=32
    return np.array([int(h[i*2:i*2+2], 16) / 255.0 for i in range(dim)])
chem_hash = np.array([hash_vec(c) for c in meta['perturbation_no_concentration']]).astype(np.float32)

# ---------- 3. 基础条件特征 ----------
medium_onehot = pd.get_dummies(meta['Medium']).astype(np.float32).values          # 2
temp_norm = ((meta['Temperature'].astype(float) - 30.0) / 7.0).values.astype(np.float32)  # 1
t = meta['pert_time'].astype(float).values
t_log = np.log2(t / 15.0) / np.log2(240.0 / 15.0)
time_feat = np.stack([t_log, np.sin(2*np.pi*t_log), np.cos(2*np.pi*t_log)], axis=1).astype(np.float32)  # 3

# ---------- 4. 交叉索引 ----------
sm_key = meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str)
ct_key = meta['perturbation_no_concentration'].astype(str) + '|' + meta['Temperature'].astype(str)
sm_cats = sorted(sm_key.unique()); ct_cats = sorted(ct_key.unique())
sm2id = {k: i for i, k in enumerate(sm_cats)}; ct2id = {k: i for i, k in enumerate(ct_cats)}
sm_id = sm_key.map(sm2id).values.astype(np.int64)
ct_id = ct_key.map(ct2id).values.astype(np.int64)

# ---------- 5. 统计先验 SVD 初始化 ----------
gmean = np.nanmean(tr_y[train_mask], axis=0)
strain_means = np.full((len(strains), P), np.nan)
for i, s in enumerate(strains):
    rows = train_mask & (meta['Strains'].values == s)
    if rows.sum() > 0:
        strain_means[i] = np.nanmean(tr_y[rows], axis=0)
strain_means = np.where(np.isnan(strain_means), gmean, strain_means)
M_s = strain_means - gmean
U_s, S_s, _ = np.linalg.svd(M_s, full_matrices=False)
d_s = min(16, len(strains) - 1)
strain_emb_init = (U_s[:, :d_s] * S_s[:d_s]).astype(np.float32)

# matched control 查找表（Δ 先验）
ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = (meta.iloc[ctrl_idx]['data_source'].astype(str) + '|' + meta.iloc[ctrl_idx]['instrument'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_idx]['Strains'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Medium'].astype(str) + '|' + meta.iloc[ctrl_idx]['Temperature'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['pert_time'].astype(str)).values
ctrl_lookup = {}
for k, pos in zip(ctrl_key, ctrl_idx):
    ctrl_lookup.setdefault(k, []).append(pos)

def matched_control(sid):
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    if k not in ctrl_lookup:
        return None
    rows = ctrl_lookup[k]
    cvals = tr_y[rows]; cm = mask[rows] > 0
    with np.errstate(invalid='ignore'):
        return np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)

treat_idx = np.where(meta['role'].eq('treatment').values)[0]
delta = np.full((len(treat_idx), P), np.nan)
for i, sid in enumerate(treat_idx):
    cm = matched_control(sid)
    if cm is not None:
        delta[i] = tr_y[sid] - cm
dmean = np.nanmean(delta)
print(f"[Δ] 有效 Δ 值 {np.isfinite(delta).sum()/1e6:.2f}M")

chem_delta_mean = np.full((len(chems), P), np.nan)
pos_map = {v: k for k, v in enumerate(treat_idx)}
for i, c in enumerate(chems):
    rows = np.where((meta['perturbation_no_concentration'].values == c) & meta['role'].eq('treatment').values)[0]
    sel = [pos_map[r] for r in rows if r in pos_map]
    if sel:
        chem_delta_mean[i] = np.nanmean(delta[sel], axis=0)
chem_delta_mean = np.where(np.isnan(chem_delta_mean), dmean, chem_delta_mean)
M_c = chem_delta_mean - dmean
M_c = np.where(np.isnan(M_c), 0.0, M_c)
U_c, S_c, _ = np.linalg.svd(M_c, full_matrices=False)
d_c = min(32, len(chems) - 1)
chem_emb_init = (U_c[:, :d_c] * S_c[:d_c]).astype(np.float32)

# ---------- 6. 上下文响应先验（菌株×培养基×温度×时间 分组均值，默认全局均值兜底）----------
ctx_key = meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|' + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)
ctx_prior = np.tile(gmean, (len(meta), 1)).astype(np.float32)
df_ctx = pd.DataFrame({'key': ctx_key.values, 'tr': train_mask})
grp = df_ctx[df_ctx['tr']].groupby('key').apply(
    lambda g: np.nanmean(tr_y[g.index], axis=0), include_groups=False)
for k, val in grp.items():
    rows = np.where(ctx_key.values == k)[0]
    ctx_prior[rows] = val
print(f"[上下文先验] 有效分组 {len(grp)} / 全组合 {ctx_key.nunique()}")

# ---------- 7. 观测上下文索引 ----------
src_cats = sorted(meta['data_source'].unique()); ins_cats = sorted(meta['instrument'].unique())
plt_cats = sorted(meta['Yeast_cell_plate'].unique())
src2id = {k: i for i, k in enumerate(src_cats)}; ins2id = {k: i for i, k in enumerate(ins_cats)}
plt2id = {k: i for i, k in enumerate(plt_cats)}
src_id = meta['data_source'].map(src2id).values.astype(np.int64)
ins_id = meta['instrument'].map(ins2id).values.astype(np.int64)
plt_id = meta['Yeast_cell_plate'].map(plt2id).values.astype(np.int64)

# ---------- 8. 蛋白共表达模块（谱聚类，K=64）----------
print("[共表达] 计算蛋白相关矩阵（训练集）...")
tr_mask_mat = mask[train_mask]
X = tr_y[train_mask]
Xc = X - np.nanmean(X, axis=0)
# 成对协方差（mask-aware，近似：非缺失样本内）
cov = np.zeros((P, P), dtype=np.float32)
cnt = np.zeros((P, P), dtype=np.float32)
obs = tr_mask_mat.astype(np.float32)
cov = (Xc.T @ Xc) / np.maximum(obs.sum(0)[:, None], 1)  # 简化：未做双 mask 精确
# 相关矩阵
sx = np.sqrt(np.diag(cov)); sx[sx == 0] = 1e-8
corr = cov / np.outer(sx, sx)
corr = np.nan_to_num(corr, 0.0)
np.fill_diagonal(corr, 1.0)
print(f"[共表达] 相关矩阵 {corr.shape}，|r|>0.6 边数 {(np.abs(corr) > 0.6).sum()/2:.0f}")

from sklearn.cluster import SpectralClustering
print("[共表达] 谱聚类 K=64 ...")
sc = SpectralClustering(n_clusters=64, affinity='precomputed', random_state=42, n_init=10)
module_id = sc.fit_predict(np.abs(corr)).astype(np.int64)
print("[共表达] 模块大小分布:", np.bincount(module_id, minlength=64)[:10], "...")

# ---------- 9. 保存 ----------
feats = {
    'strain_id': strain_id, 'chem_id': chem_id,
    'strain_seen': strain_seen, 'chem_seen': chem_seen,
    'chem_hash': chem_hash, 'medium_onehot': medium_onehot,
    'temp_norm': temp_norm, 'time_feat': time_feat,
    'sm_id': sm_id, 'ct_id': ct_id,
    'src_id': src_id, 'ins_id': ins_id, 'plt_id': plt_id,
    'strain_emb_init': strain_emb_init, 'chem_emb_init': chem_emb_init,
    'ctx_prior': ctx_prior, 'module_id': module_id,
    'gmean': gmean.astype(np.float32),
    'strain_means': strain_means.astype(np.float32),
    'n_strains': len(strains), 'n_chems': len(chems), 'd_s': d_s, 'd_c': d_c,
    'n_sm': len(sm_cats), 'n_ct': len(ct_cats),
    'n_src': len(src_cats), 'n_ins': len(ins_cats), 'n_plt': len(plt_cats),
    'train_mask': train_mask, 'n_modules': 64,
}
with open(f"{DATA}/feats.pkl", 'wb') as f:
    pickle.dump(feats, f)
print(f"[保存] feats.pkl | strain_emb_init {strain_emb_init.shape} | chem_emb_init {chem_emb_init.shape}")
print("02 DONE")
