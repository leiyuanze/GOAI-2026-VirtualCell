# -*- coding: utf-8 -*-
"""
步骤12 前置：可靠性门控特征（gpt2 P1-4）
- chem_max_sim：样本化合物与训练化合物的最大 Morgan 相似度（test 用 train 池算，合规）
- chem_support：化合物在训练集处理样本数（log1p）
- strain_support：菌株在训练集行数（log1p）
写入 feats['chem_max_sim'] / feats['chem_support'] / feats['strain_support']
+ feats['test_chem_max_sim'] / feats['test_chem_support']（test 行）
"""
import pickle
import numpy as np
import pandas as pd

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
BASE = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"

meta = pd.read_pickle(f"{DATA}/meta.pkl")
tmeta = pd.read_csv(f"{BASE}/WAYB_WAYC_metadata_test(1).csv").set_index('sample_ID')
feats = pickle.load(open(f"{DATA}/feats.pkl", 'rb'))
train_mask = meta['split_final'].eq('train').values
pert2morgan = feats['pert2morgan64']

train_chems = sorted(meta.loc[train_mask & meta['role'].eq('treatment'), 'perturbation_no_concentration'])
train_strains = sorted(meta.loc[train_mask, 'Strains'])
train_chem_vectors = np.stack([pert2morgan[c] for c in train_chems])  # (n_chem, 64)

def max_sim(v):
    if v is None or np.linalg.norm(v) == 0:
        return 0.0
    sims = train_chem_vectors @ v / (np.linalg.norm(train_chem_vectors, axis=1) * np.linalg.norm(v) + 1e-8)
    return float(sims.max())

# 训练支持数
chem_support = {c: 0 for c in train_chems}
strain_support = {s: 0 for s in train_strains}
pert_all = meta['perturbation_no_concentration'].values
strain_all = meta['Strains'].values
for i in np.where(train_mask)[0]:
    if pert_all[i] in chem_support:
        chem_support[pert_all[i]] += 1
    if strain_all[i] in strain_support:
        strain_support[strain_all[i]] += 1

# train_val 样本
N = len(meta)
cms = np.zeros(N, dtype=np.float32)
csp = np.zeros(N, dtype=np.float32)
ssp = np.zeros(N, dtype=np.float32)
for i in range(N):
    c = pert_all[i]
    v = pert2morgan.get(c)
    cms[i] = max_sim(v) if c not in train_chems else 1.0  # 已见化合物相似度=1
    csp[i] = np.log1p(chem_support.get(c, 0))
    ssp[i] = np.log1p(strain_support.get(strain_all[i], 0))

# test 样本
tc = tmeta['perturbation_no_concentration'].values
ts = tmeta['Strains'].values
tcms = np.array([max_sim(pert2morgan.get(c)) if c not in train_chems else 1.0 for c in tc], dtype=np.float32)
tcsp = np.array([np.log1p(chem_support.get(c, 0)) for c in tc], dtype=np.float32)
tssp = np.array([np.log1p(strain_support.get(s, 0)) for s in ts], dtype=np.float32)

feats['chem_max_sim'] = cms
feats['chem_support'] = csp
feats['strain_support'] = ssp
feats['test_chem_max_sim'] = tcms
feats['test_chem_support'] = tcsp
feats['test_strain_support'] = tssp
with open(f"{DATA}/feats.pkl", 'wb') as f:
    pickle.dump(feats, f)
print(f"chem_max_sim: {cms.shape} 分布 [{cms.min():.3f}, {cms.max():.3f}]")
print(f"chem_support: {csp.shape} 分布 [{csp.min():.3f}, {csp.max():.3f}]")
print(f"strain_support: {ssp.shape} 分布 [{ssp.min():.3f}, {ssp.max():.3f}]")
print("RELIABILITY FEATS DONE")
