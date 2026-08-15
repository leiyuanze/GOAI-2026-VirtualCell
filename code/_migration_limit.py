# -*- coding: utf-8 -*-
"""检验化学结构迁移的理论上限：Tamoxifen(test) vs 4-OH-Tamoxifen(train)"""
import numpy as np, pandas as pd, pickle

DATA = 'data'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float64)
mask = np.load(f'{DATA}/mask.npy').astype(bool)
tr_y_nan = np.where(mask, y_log2, np.nan)
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))
tmeta = pd.read_csv('../input/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')
traw = pd.read_csv('../input/WAYB_WAYC_proteome_raw_test.csv').set_index('sample_ID')
P = y_log2.shape[1]

def mk_key(df):
    return (df['data_source'].astype(str) + '|' + df['instrument'].astype(str) + '|'
            + df['Yeast_cell_plate'].astype(str) + '|' + df['Strains'].astype(str) + '|'
            + df['Medium'].astype(str) + '|' + df['Temperature'].astype(str) + '|'
            + df['pert_time'].astype(str)).values

# 训练集对照
ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = mk_key(meta.iloc[ctrl_idx])
ctrl_lookup = {}
for k, pos in zip(ctrl_key, ctrl_idx):
    ctrl_lookup.setdefault(k, []).append(pos)
treat_all = np.where(meta['role'].eq('treatment').values)[0]
ctrl_all = np.full((len(treat_all), P), np.nan)
for i, sid in enumerate(treat_all):
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    rows = ctrl_lookup.get(k, [])
    if not rows:
        continue
    cvals = tr_y_nan[rows]; cm = mask[rows]
    with np.errstate(invalid='ignore'):
        ctrl_all[i] = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)
delta_tr = tr_y_nan[treat_all] - ctrl_all
chem_of = meta['perturbation_no_concentration'].values[treat_all]
train_mask = meta['split_final'].eq('train').values
is_train = np.isin(treat_all, np.where(train_mask & meta['role'].eq('treatment').values)[0])

# 每个训练化合物的平均 Δ（μ_drug）
drug_mu = {}
for c in set(chem_of[is_train]):
    m = np.isin(chem_of, c) & is_train
    drug_mu[c] = np.nanmean(delta_tr[m], axis=0)

# test_chem_only 的 Tamoxifen
idx = np.where(tmeta['split_final'].eq('test_chem_only').values)[0]
treat_pos = [p for p in idx if not tmeta.iloc[p]['perturbation_no_concentration'] in ('Water', 'DMSO')]
t_log2 = np.log2(traw.iloc[treat_pos].values.astype(np.float64))
test_cols = traw.columns.tolist()
prot4422 = [l.strip() for l in open(f'{DATA}/prot_names.txt')]
col_of = {p: i for i, p in enumerate(test_cols)}
pos4422 = np.array([col_of[p] for p in prot4422], dtype=int)
t_key = mk_key(tmeta)
ctrl_te = {}
for k, pos in zip(t_key, np.where(tmeta['perturbation_no_concentration'].isin(['Water', 'DMSO']).values)[0]):
    ctrl_te.setdefault(k, []).append(pos)

# 用指纹相似度选最相似训练化合物，对每个 test 样本做迁移预测
pert2morgan = feats['pert2morgan64']
train_chems = [c for c in drug_mu if c in pert2morgan]

def best_train_chem(test_c):
    v = pert2morgan.get(test_c)
    if v is None:
        return None
    best_c, best_s = None, -1
    for tc in train_chems:
        w = pert2morgan[tc]
        s = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-8)
        if s > best_s:
            best_s, best_c = s, tc
    return best_c, best_s

# 检验每个 test 新化合物用 best 迁移的 PCC
print(f"{'化合物':<28}{'最近邻':<24}{'sim':>6}{'迁移PCC':>9}{'n':>6}")
t_chems = sorted(set(tmeta.iloc[treat_pos]['perturbation_no_concentration']))
for c in t_chems:
    if c in drug_mu:  # 训练见过，跳过
        continue
    bt = best_train_chem(c)
    if bt is None:
        continue
    tc, sim = bt
    # test 该化合物的真值 Δ
    samp = [i for i, p in enumerate(treat_pos) if tmeta.iloc[p]['perturbation_no_concentration'] == c]
    if not samp:
        continue
    # 用第一个样本匹配对照
    rows = ctrl_te.get(t_key[treat_pos[samp[0]]], [])
    if not rows:
        continue
    yc = np.nanmean(t_log2[rows], axis=0)
    dt_all = t_log2[samp] - yc
    pred = np.full(len(test_cols), np.nan)
    pred[pos4422] = drug_mu[tc]
    ok = np.isfinite(dt_all) & np.isfinite(pred)
    if ok.sum() <= 10:
        print(f"{c:<28}{tc:<24}{sim:>6.3f}{'nan':>9}{ok.sum():>6}")
        continue
    pcc = np.corrcoef(dt_all[ok].ravel(), np.tile(pred, (len(samp), 1))[ok].ravel())[0, 1]
    print(f"{c:<28}{tc:<24}{sim:>6.3f}{pcc:>9.3f}{ok.sum():>6}")
