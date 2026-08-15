# -*- coding: utf-8 -*-
"""验证菌株遗传迁移对 unseen 菌株 BAI 的实际收益"""
import gzip, numpy as np, pandas as pd, pickle

meta = pd.read_pickle('data/meta.pkl')
y = np.load('data/y_log2.npy'); mask = np.load('data/mask.npy').astype(bool)
feats = pickle.load(open('data/feats.pkl','rb'))
gmean = feats['gmean']  # 4422
strain_means = feats['strain_means']  # (5, 4422) 对应 5 个菌株
strains = sorted(meta['Strains'].unique())  # BAH, BAI, CEK, CGD, DHY210

# 遗传距离（从 1011 矩阵）
D = {'BAH': {'BAH':0,'BAI':1.775,'CEK':2.105,'CGD':2.218,'CRD':2.099},
     'BAI': {'BAH':1.775,'BAI':0,'CEK':1.362,'CGD':1.607,'CRD':1.478},
     'CEK': {'BAH':2.105,'BAI':1.362,'CEK':0,'CGD':1.609,'CRD':1.475},
     'CGD': {'BAH':2.218,'BAI':1.607,'CEK':1.609,'CGD':0,'CRD':0.398},
     'CRD': {'BAH':2.099,'BAI':1.478,'CEK':1.475,'CGD':0.398,'CRD':0}}
# DHY210 用 S288C 代理，距离设为 2.3（远）
for s in D: D[s]['DHY210'] = 2.3
D['DHY210'] = {s: 2.3 for s in ['BAH','BAI','CEK','CGD','CRD','DHY210']}

train_strains = ['BAH','CEK','CGD','DHY210']

def sim(s, t, scale=1.0):
    return np.exp(-D[s][t] / scale)

# BAI 的遗传加权特异响应
s2i = {s: i for i, s in enumerate(strains)}
ba_idx = s2i['BAI']
# 训练菌株的特异响应
tr_specific = {}
for t in train_strains:
    ti = s2i[t]
    tr_specific[t] = strain_means[ti] - gmean

# 遗传加权
w = np.array([sim('BAI', t) for t in train_strains])
w = w / w.sum()
pred_genetic = sum(w[i] * tr_specific[train_strains[i]] for i in range(4))
# 平均（现有模型的做法，等价于 strain_avg - gmean）
pred_avg = np.mean(np.stack(list(tr_specific.values())), axis=0)

# BAI 的真实特异响应（val 里 BAI 的处理样本均值 - gmean）
ctrl_rows = meta['split_final'].eq('train').values & meta['role'].eq('control').values
ctrl_mean = np.nanmean(np.where(mask[ctrl_rows], y[ctrl_rows], np.nan), axis=0)
bai_rows = meta['Strains'].eq('BAI').values & meta['role'].eq('treatment').values
bai_true = np.nanmean(np.where(mask[bai_rows], y[bai_rows], np.nan), axis=0) - ctrl_mean

def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0,1]) if ok.sum() > 100 else float('nan')

r_genetic = corr(pred_genetic, bai_true)
r_avg = corr(pred_avg, bai_true)
print(f'BAI 遗传相似度权重: ' + ', '.join(f'{t}={w[i]:.3f}' for i,t in enumerate(train_strains)))
print(f'遗传加权预测 vs BAI真实响应 相关系数: {r_genetic:.4f}')
print(f'平均预测(现有做法) vs BAI真实响应 相关系数: {r_avg:.4f}')
print(f'提升: {r_genetic - r_avg:+.4f}')
print()
# CRD 的遗传权重（test 无法验证，但看结构）
w_crd = np.array([sim('CRD', t) for t in train_strains]); w_crd = w_crd / w_crd.sum()
print(f'CRD 遗传相似度权重: ' + ', '.join(f'{t}={w_crd[i]:.3f}' for i,t in enumerate(train_strains)))
print('   -> CRD 主要由 CGD 主导' if w_crd[train_strains.index("CGD")] > 0.5 else '')
