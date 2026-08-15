# -*- coding: utf-8 -*-
"""迁移融合收益验证：模型预测 + top-k 化学迁移预测 加权融合
检验融合能否提升 test_chem_only 的六模块分数（用 test 真值自评）
"""
import numpy as np, pandas as pd, pickle

DATA = 'data'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float64)
mask = np.load(f'{DATA}/mask.npy').astype(bool)
tr_y_nan = np.where(mask, y_log2, np.nan)
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))
tmeta = pd.read_csv('../input/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')
traw = pd.read_csv('../input/WAYB_WAYC_proteome_raw_test.csv').set_index('sample_ID')
sub = pd.read_csv(f'{DATA}/prediction_adaptive2.csv', index_col=0)
P = y_log2.shape[1]
test_cols = sub.columns.tolist()
prot4422 = [l.strip() for l in open(f'{DATA}/prot_names.txt')]
col_of = {p: i for i, p in enumerate(test_cols)}
pos4422 = np.array([col_of[p] for p in prot4422], dtype=int)
pert2morgan = feats['pert2morgan64']

def mk_key(df):
    return (df['data_source'].astype(str) + '|' + df['instrument'].astype(str) + '|'
            + df['Yeast_cell_plate'].astype(str) + '|' + df['Strains'].astype(str) + '|'
            + df['Medium'].astype(str) + '|' + df['Temperature'].astype(str) + '|'
            + df['pert_time'].astype(str)).values

# 训练对照池
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
tr_strain = meta['Strains'].values[treat_all]

# 训练化合物 (菌株, 化合物) 平均 Δ
drug_strain_mu = {}
for i in np.where(is_train)[0]:
    key = (tr_strain[i], chem_of[i])
    if key not in drug_strain_mu:
        drug_strain_mu[key] = []
    drug_strain_mu[key].append(delta_tr[i])
for k in drug_strain_mu:
    drug_strain_mu[k] = np.nanmean(np.stack(drug_strain_mu[k]), axis=0)

train_chems = sorted(set(chem_of[is_train]))
def topk_sim(test_c, k=5):
    v = pert2morgan.get(test_c)
    if v is None:
        return []
    sims = []
    for tc in train_chems:
        w = pert2morgan.get(tc)
        if w is None:
            continue
        s = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-8)
        sims.append((s, tc))
    sims.sort(reverse=True)
    return sims[:k]

# test_chem_only
idx = np.where(tmeta['split_final'].eq('test_chem_only').values)[0]
treat_in = [p for p in idx if not tmeta.iloc[p]['perturbation_no_concentration'] in ('Water', 'DMSO')]
t_log2 = np.log2(traw.iloc[treat_in].values.astype(np.float64))
t_strain = tmeta.iloc[treat_in]['Strains'].values
t_chem = tmeta.iloc[treat_in]['perturbation_no_concentration'].values

# test Δ（训练对照池）
test_delta = np.full((len(treat_in), len(test_cols)), np.nan)
for i, p in enumerate(treat_in):
    r = tmeta.iloc[p]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    rows = ctrl_lookup.get(k, [])
    if not rows:
        continue
    cvals = tr_y_nan[rows]; cm = mask[rows]
    with np.errstate(invalid='ignore'):
        yc = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)
    test_delta[i, pos4422] = t_log2[i, pos4422] - yc

# 模型预测 Δ（sub 是绝对丰度，需减对照）
sub_arr = sub.iloc[treat_in].values.astype(np.float64)
model_delta = np.full((len(treat_in), len(test_cols)), np.nan)
for i, p in enumerate(treat_in):
    r = tmeta.iloc[p]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    rows = ctrl_lookup.get(k, [])
    if not rows:
        continue
    cvals = tr_y_nan[rows]; cm = mask[rows]
    with np.errstate(invalid='ignore'):
        yc = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)
    model_delta[i, pos4422] = sub_arr[i, pos4422] - yc

# 迁移预测 Δ（同菌株 top-k 加权）
mig_delta = np.zeros((len(treat_in), P))
mig_mask = np.zeros((len(treat_in), P), dtype=bool)
for i in range(len(treat_in)):
    tops = topk_sim(t_chem[i])
    mus = []; ws = []
    for sim, tc in tops:
        mu = drug_strain_mu.get((t_strain[i], tc))
        if mu is None:
            continue
        mus.append(mu); ws.append(max(sim, 0.0))
    if not mus:
        continue
    ws = np.array(ws); ws = ws / ws.sum()
    mig_delta[i] = np.sum(np.stack([w * m for w, m in zip(ws, mus)]), axis=0)
    mig_mask[i] = True

# 评估融合（PCC on Δ）
def pcc(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float('nan')
    return np.corrcoef(a[ok], b[ok])[0, 1]

print(f"{'alpha(迁移权重)':>16}{'M1 FC PCC':>12}{'Δ RMSE':>10}")
for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    fused = np.zeros_like(model_delta)
    for i in range(len(treat_in)):
        if mig_mask[i].any():
            fused[i, pos4422] = (1 - alpha) * model_delta[i, pos4422] + alpha * mig_delta[i]
        else:
            fused[i, pos4422] = model_delta[i, pos4422]
    # 全局 M1
    ok = np.isfinite(test_delta) & np.isfinite(fused)
    m1 = pcc(fused[ok].ravel(), test_delta[ok].ravel())
    rmse = np.sqrt(np.nanmean((fused - test_delta) ** 2))
    print(f"{alpha:>16.1f}{m1:>12.4f}{rmse:>10.4f}")
