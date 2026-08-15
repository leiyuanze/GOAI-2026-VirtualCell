# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 14 菌株遗传相似度特征（v3.3）
从 1011 项目 SNP 距离矩阵，计算每个样本的菌株与训练菌株的遗传相似度。
- seen 菌株：one-hot（自己=1）
- unseen 菌株（BAI/CRD）：遗传相似度（exp(-D) 归一化）
输出：feats['strain_sim']（N x 4，对应训练菌株 BAH/CEK/CGD/DHY210）
外部数据来源：1011 Yeast Genomes Project (Peter et al. 2018), SNP 距离矩阵
"""
import numpy as np
import pandas as pd
import pickle

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"

meta = pd.read_pickle(f"{DATA}/meta.pkl")
feats = pickle.load(open(f"{DATA}/feats.pkl", 'rb'))
N = len(meta)

# 6 菌株遗传距离（从 1011 SNP 距离矩阵提取）
D = {'BAH': {'BAH':0,'BAI':1.775,'CEK':2.105,'CGD':2.218,'CRD':2.099},
     'BAI': {'BAH':1.775,'BAI':0,'CEK':1.362,'CGD':1.607,'CRD':1.478},
     'CEK': {'BAH':2.105,'BAI':1.362,'CEK':0,'CGD':1.609,'CRD':1.475},
     'CGD': {'BAH':2.218,'BAI':1.607,'CEK':1.609,'CGD':0,'CRD':0.398},
     'CRD': {'BAH':2.099,'BAI':1.478,'CEK':1.475,'CGD':0.398,'CRD':0}}
# DHY210 是 S288C 实验室背景，不在 1011 野生菌株矩阵，距离设为 2.3（远）
for s in D:
    D[s]['DHY210'] = 2.3
D['DHY210'] = {s: 2.3 for s in ['BAH','BAI','CEK','CGD','CRD','DHY210']}

train_strains = ['BAH', 'CEK', 'CGD', 'DHY210']  # 训练菌株（字母序）

def sim(s, t):
    return np.exp(-D[s][t])

# 每个样本的遗传相似度向量
strain_of = meta['Strains'].values
strain_sim = np.zeros((N, len(train_strains)), dtype=np.float32)
for i, s in enumerate(strain_of):
    if s in train_strains:  # seen 菌株：one-hot
        strain_sim[i, train_strains.index(s)] = 1.0
    else:  # unseen 菌株（BAI/CRD）：遗传相似度
        w = np.array([sim(s, t) for t in train_strains])
        w = w / w.sum()
        strain_sim[i] = w.astype(np.float32)

feats['strain_sim'] = strain_sim
feats['n_train_strains'] = len(train_strains)

# ★ 遗传加权基线：对 unseen 菌株，ctx_prior 用遗传加权菌株均值替代 gmean 兜底
strains_all = sorted(meta['Strains'].unique())
s2i_all = {s: i for i, s in enumerate(strains_all)}
gmean = feats['gmean']
strain_means = feats['strain_means']  # (5, 4422)
train_idx = [s2i_all[t] for t in train_strains]
n_updated = 0
for i, s in enumerate(strain_of):
    if s not in train_strains:  # unseen 菌株（BAI）
        w = strain_sim[i]
        weighted = gmean.copy()
        for j in range(len(train_strains)):
            weighted = weighted + w[j] * (strain_means[train_idx[j]] - gmean)
        feats['ctx_prior'][i] = weighted.astype(np.float32)
        n_updated += 1
print(f'遗传加权基线更新 {n_updated} 个 unseen 菌株样本的 ctx_prior')

with open(f"{DATA}/feats.pkl", 'wb') as f:
    pickle.dump(feats, f)

# 验证
print(f"strain_sim: {strain_sim.shape}")
for s in ['BAH', 'BAI', 'CRD', 'CGD']:
    rows = np.where(strain_of == s)[0]
    if len(rows):
        print(f"  {s}: {strain_sim[rows[0]]}")
print("14 DONE")
