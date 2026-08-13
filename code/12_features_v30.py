# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 12 外部特征集成（v3.0）
1. Morgan 指纹（2048维）-> feats['chem_morgan']（N x 2048，按样本）
2. ESM2 embedding（320维）-> PCA 到 64 维 -> feats['esm2_emb']（4422 x 64）
输出：更新 feats.pkl
外部数据来源：PubChem(RDKit Morgan fingerprint), ESM2(Meta)
"""
import pickle
import numpy as np
import pandas as pd

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"

meta = pd.read_pickle(f"{DATA}/meta.pkl")
feats = pickle.load(open(f"{DATA}/feats.pkl", 'rb'))
N = len(meta)

# ---------- 1. Morgan 指纹 ----------
fp_map = pickle.load(open(f"{DATA}/chem_morgan.pkl", 'rb'))
# 对齐：meta 的 perturbation 名 -> 指纹
pert = meta['perturbation_no_concentration'].values
n_missing = 0
chem_morgan = np.zeros((N, 2048), dtype=np.float32)
for i, p in enumerate(pert):
    if p in fp_map:
        chem_morgan[i] = fp_map[p]
    else:
        n_missing += 1  # 用零向量
print(f"[1] Morgan 指纹原始: {chem_morgan.shape}, 缺失 {n_missing} 样本(零向量)")

# PCA 降维到 64（46 个 unique 指纹，PCA 到 64 保留主要结构差异）
morgan_c = chem_morgan - chem_morgan.mean(axis=0)
U, S, _ = np.linalg.svd(morgan_c, full_matrices=False)
morgan_64 = (U[:, :64] * S[:64]).astype(np.float32)
# 归一化（防止量纲过大）
morgan_64 = morgan_64 / (np.linalg.norm(morgan_64, axis=1, keepdims=True) + 1e-6)
print(f"    PCA 降维 -> {morgan_64.shape}, 方差解释 {S[:64].sum()/S.sum()*100:.1f}%")
feats['chem_morgan'] = morgan_64

# ---------- 2. ESM2 embedding PCA 降维 ----------
esm2 = np.load(f"{DATA}/prot_esm2.npy").astype(np.float32)  # 4422 x 320
# 标准化后 SVD
esm2_c = esm2 - esm2.mean(axis=0)
U, S, Vt = np.linalg.svd(esm2_c, full_matrices=False)
d = 64
esm2_emb = (U[:, :d] * S[:d]).astype(np.float32)  # 4422 x 64
# 归一化（防止量纲过大）
esm2_emb = esm2_emb / (np.linalg.norm(esm2_emb, axis=1, keepdims=True) + 1e-6)
print(f"[2] ESM2: {esm2.shape} -> PCA {esm2_emb.shape}, 方差解释 {S[:d].sum()/S.sum()*100:.1f}%")
feats['esm2_emb'] = esm2_emb

# ---------- 3. 保存 ----------
with open(f"{DATA}/feats.pkl", 'wb') as f:
    pickle.dump(feats, f)
print("[3] feats.pkl 已更新，新增键: chem_morgan(2048), esm2_emb(64)")
print("12 DONE")
