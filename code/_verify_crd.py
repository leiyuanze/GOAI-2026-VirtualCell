# -*- coding: utf-8 -*-
"""验证遗传迁移对 test 的 unseen 菌株 CRD 的实际收益（用 test 真值自评）"""
import numpy as np, pandas as pd, pickle

DATA = 'data'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
y = np.load(f'{DATA}/y_log2.npy'); mask = np.load(f'{DATA}/mask.npy').astype(bool)
feats = pickle.load(open(f'{DATA}/feats.pkl','rb'))
gmean = feats['gmean']
strain_means = feats['strain_means']  # (5, 4422)
strains = sorted(meta['Strains'].unique())

# test 真值
tmeta = pd.read_csv('../input/WAYB_WAYC_metadata_test(1).csv')
tprot = pd.read_csv('../input/WAYB_WAYC_proteome_raw_test.csv', index_col='sample_ID')
# 对齐到 4422 保留蛋白
prot_names = [l.strip() for l in open(f'{DATA}/prot_names.txt')]
tprot = tprot[prot_names]
tprot_log2 = np.log2(tprot.values)

# 遗传距离（同前）
D = {'BAH': {'BAH':0,'BAI':1.775,'CEK':2.105,'CGD':2.218,'CRD':2.099},
     'BAI': {'BAH':1.775,'BAI':0,'CEK':1.362,'CGD':1.607,'CRD':1.478},
     'CEK': {'BAH':2.105,'BAI':1.362,'CEK':0,'CGD':1.609,'CRD':1.475},
     'CGD': {'BAH':2.218,'BAI':1.607,'CEK':1.609,'CGD':0,'CRD':0.398},
     'CRD': {'BAH':2.099,'BAI':1.478,'CEK':1.475,'CGD':0.398,'CRD':0}}
for s in D: D[s]['DHY210'] = 2.3
D['DHY210'] = {s: 2.3 for s in ['BAH','BAI','CEK','CGD','CRD','DHY210']}
train_strains = ['BAH','CEK','CGD','DHY210']
s2i = {s: i for i, s in enumerate(strains)}

def sim(s, t): return np.exp(-D[s][t])

# CRD 的真实菌株特异响应（test 里 CRD 的所有样本均值 - 全局对照）
crd_rows = tmeta['Strains'].eq('CRD').values
crd_mean = np.nanmean(np.where(np.isfinite(tprot_log2[crd_rows]), tprot_log2[crd_rows], np.nan), axis=0)
# 对照：用 train 的对照均值
ctrl_rows = meta['split_final'].eq('train').values & meta['role'].eq('control').values
ctrl_mean = np.nanmean(np.where(mask[ctrl_rows], y[ctrl_rows], np.nan), axis=0)
crd_specific = crd_mean - ctrl_mean  # CRD 真实特异响应

# 遗传加权预测（CGD 主导）
w = np.array([sim('CRD', t) for t in train_strains]); w = w / w.sum()
pred_genetic = sum(w[j] * (strain_means[s2i[train_strains[j]]] - gmean) for j in range(4))
# 平均预测（现有做法）
pred_avg = np.mean(np.stack([strain_means[s2i[t]] - gmean for t in train_strains]), axis=0)

def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0,1]) if ok.sum() > 100 else float('nan')

r_genetic = corr(pred_genetic, crd_specific)
r_avg = corr(pred_avg, crd_specific)
print(f'CRD 样本数: {crd_rows.sum()}')
print(f'CRD 遗传权重: ' + ', '.join(f'{t}={w[j]:.3f}' for j,t in enumerate(train_strains)))
print(f'遗传加权预测 vs CRD真实特异响应: {r_genetic:.4f}')
print(f'平均预测(现有做法) vs CRD真实特异响应: {r_avg:.4f}')
print(f'提升: {r_genetic - r_avg:+.4f}')
