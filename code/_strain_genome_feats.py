# -*- coding: utf-8 -*-
"""
步骤11：菌株功能特征 —— SNP 遗传距离向量（gpt2 第五阶段 / P2-1 第一优先级变体）
用 1011 项目 SNP 距离矩阵，为每个菌株构建「到训练菌株的遗传距离向量」：
  strain_dist_vec（N×4）：每样本菌株到 4 个训练划分菌株（BAH/CEK/CGD/DHY210）的 SNP 距离
  - unseen 菌株（BAI val / CRD test）也有真实遗传距离 → 模型可学习"遗传近的菌株响应可迁移"
  - DHY210 缺失：用其与其余 4 菌株的平均距离（保守等距代理）
外部数据来源：1011 酵母基因组项目 SNP 距离（开放榜允许，已披露）
"""
import gzip
import numpy as np
import pandas as pd
import pickle

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
BASE = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"

meta = pd.read_pickle(f"{DATA}/meta.pkl")
tmeta = pd.read_csv(f"{BASE}/WAYB_WAYC_metadata_test(1).csv").set_index('sample_ID')
feats = pickle.load(open(f"{DATA}/feats.pkl", 'rb'))
train_mask = meta['split_final'].eq('train').values

# 训练划分菌株（BAI 只在 val）
train_strains = sorted(meta.loc[train_mask, 'Strains'].unique())
print(f"训练划分菌株: {train_strains}")
all_strains = sorted(set(meta['Strains'].unique()) | set(tmeta['Strains'].unique()))
print(f"全菌株: {all_strains}")

# 读 SNP 距离矩阵
with gzip.open(f"{DATA}/1011_SNP_distance.tab.gz", 'rt') as f:
    names_all = f.readline().strip().split('\t')
    dist = np.loadtxt(f, usecols=range(1, len(names_all)))  # 跳过菌株名列
n_cols = dist.shape[1]
names = names_all[:n_cols]  # 列对齐（header 可能多一个名字）
name2i = {n: i for i, n in enumerate(names)}
print(f"SNP 矩阵 {dist.shape}（列对齐到 {len(names)} 菌株）")

# 提取目标菌株子矩阵
def get_dist(s):
    if s in name2i:
        return dist[name2i[s]]
    return None

# DHY210 缺失 → 用其余目标菌株的平均距离向量代理
present = {s: get_dist(s) for s in all_strains}
missing = [s for s in all_strains if present[s] is None]
print(f"SNP 缺失菌株: {missing}")
if 'DHY210' in missing:
    others = [present[s] for s in all_strains if present[s] is not None]
    present['DHY210'] = np.mean(others, axis=0)

# 构建菌株→距离向量（到训练菌株）
strain_vec = {}
for s in all_strains:
    vec = np.array([present[s][name2i[ts]] if ts in name2i else np.nan for ts in train_strains], dtype=np.float32)
    # 自身距离=0
    for ts in train_strains:
        if s == ts and ts in name2i:
            vec[train_strains.index(ts)] = 0.0
    # 缺失列（如 DHY210 代理中无自身）用中位距离
    vec = np.where(np.isnan(vec), np.nanmedian(present[s][list(name2i.values())]), vec)
    strain_vec[s] = vec.astype(np.float32)

print("菌株距离向量（到训练菌株）:")
for s in all_strains:
    print(f"  {s}: {np.round(strain_vec[s], 3)}")

# 样本级特征（train_val + test）
sdv = np.stack([strain_vec[s] for s in meta['Strains'].values]).astype(np.float32)
tsdv = np.stack([strain_vec[s] for s in tmeta['Strains'].values]).astype(np.float32)
print(f"\nstrain_dist_vec: {sdv.shape} | test: {tsdv.shape}")

# 可选：距离 → 相似度核（高斯核，σ=中位距离）
med = np.median(sdv)
K = np.exp(-(sdv ** 2) / (2 * med ** 2)).astype(np.float32)
tK = np.exp(-(tsdv ** 2) / (2 * med ** 2)).astype(np.float32)

feats['strain_dist_vec'] = sdv
feats['test_strain_dist_vec'] = tsdv
feats['strain_kernel'] = K
feats['test_strain_kernel'] = tK
feats['train_strains_genome'] = train_strains
with open(f"{DATA}/feats.pkl", 'wb') as f:
    pickle.dump(feats, f)
print("\nfeats.pkl 已更新: strain_dist_vec + strain_kernel")
print("STEP11 DONE")
