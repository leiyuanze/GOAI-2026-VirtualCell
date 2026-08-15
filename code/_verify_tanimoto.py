# -*- coding: utf-8 -*-
"""验证 Tanimoto 相似度迁移对 unseen 化合物的实际收益"""
import pickle, numpy as np, pandas as pd

fp_map = pickle.load(open('data/chem_morgan.pkl','rb'))
meta = pd.read_pickle('data/meta.pkl')
y = np.load('data/y_log2.npy'); mask = np.load('data/mask.npy').astype(bool)
tr = meta['split_final'].eq('train').values

def tanimoto(a, b):
    a = a.astype(bool); b = b.astype(bool)
    return (a & b).sum() / max((a | b).sum(), 1)

train_chems = sorted(meta.loc[tr & meta['role'].eq('treatment'), 'perturbation_no_concentration'].unique())
ctrl_rows = tr & meta['role'].eq('control').values
ctrl_mean = np.nanmean(np.where(mask[ctrl_rows], y[ctrl_rows], np.nan), axis=0)

chem_delta = {}
for c in train_chems:
    rows = tr & meta['role'].eq('treatment').values & (meta['perturbation_no_concentration'] == c).values
    cm = np.nanmean(np.where(mask[rows], y[rows], np.nan), axis=0)
    chem_delta[c] = cm - ctrl_mean

unseen = ['Amphotericin B','FCCP','Hydroxyurea','Pentamidine isethionate','Raloxifene hydrochloride','Sulfometuron methyl']

def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0,1]) if ok.sum() > 10 else float('nan')

print(f"{'compound':<30}{'top_neighbor':<22}{'sim':>7}{'weightedR':>11}{'avgR':>10}")
print('-' * 82)
for c in unseen:
    fp = fp_map.get(c, np.zeros(2048))
    sims = np.array([tanimoto(fp, fp_map.get(tc, np.zeros(2048))) for tc in train_chems])
    w = sims / max(sims.sum(), 1e-9)
    pred_w = sum(w[i] * chem_delta[train_chems[i]] for i in range(len(train_chems)))
    pred_avg = np.nanmean(np.stack(list(chem_delta.values())), axis=0)
    vrows = meta['role'].eq('treatment').values & (meta['perturbation_no_concentration'] == c).values & ~tr
    true_d = np.nanmean(np.where(mask[vrows], y[vrows], np.nan), axis=0) - ctrl_mean
    rw = corr(pred_w, true_d); ra = corr(pred_avg, true_d)
    top = train_chems[int(sims.argmax())]
    print(f"{c:<30}{top[:22]:<22}{sims.max():>7.3f}{rw:>11.3f}{ra:>10.3f}")

# 汇总：Tanimoto 迁移整体 vs 平均迁移
print()
print('=' * 82)
print('结论：weightedR 是 Tanimoto 加权迁移与真实响应的相关，avgR 是平均迁移（无外部信息）的相关')
print('      weightedR > avgR 说明 Tanimoto 迁移有效')
