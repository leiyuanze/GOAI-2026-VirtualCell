# -*- coding: utf-8 -*-
"""迁移融合最终提交：模型预测 + 化学结构迁移预测 加权融合
alpha=0.2（由 _mig_fusion 网格搜索确定，M1 +0.006 / RMSE -0.011）
仅对 test_chem_only / test_both（新化合物场景）融合；strain/time 用原模型
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
ALPHA = 0.2
NEW_CHEM_SCENES = ('test_chem_only', 'test_both')

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

def matched_control_mean(rows_src, src):
    cvals = src[rows_src]; cm = np.isfinite(cvals)
    with np.errstate(invalid='ignore'):
        return np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, 0), 0) / cm.sum(0), np.nan)

# 对每个 test 样本计算迁移 Δ（绝对丰度空间：预测的 y_control + 迁移 Δ）
# 迁移 Δ 是 4422 维（训练蛋白），映射回 5243
new = sub.copy().values.astype(np.float64)
is_new_chem = tmeta['perturbation_no_concentration'].isin(
    [c for c in tmeta['perturbation_no_concentration'].unique() if c not in set(chem_of)]
).values

for i in range(len(tmeta)):
    if not is_new_chem[i]:
        continue
    r = tmeta.iloc[i]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    rows = ctrl_lookup.get(k, [])
    if not rows:
        continue
    yc = matched_control_mean(rows, tr_y_nan)  # 4422 维对照
    if not np.isfinite(yc).any():
        continue
    c = r['perturbation_no_concentration']
    tops = topk_sim(c)
    mus = []; ws = []
    for sim, tc in tops:
        mu = drug_strain_mu.get((r['Strains'], tc))
        if mu is None:
            continue
        mus.append(mu); ws.append(max(sim, 0.0))
    if not mus:
        continue
    ws = np.array(ws); ws = ws / ws.sum()
    mig_delta = np.sum(np.stack([w * m for w, m in zip(ws, mus)]), axis=0)  # 4422
    # 融合：只对 yc 有值且迁移有值的蛋白做
    model_delta = new[i, pos4422] - yc
    fused = yc + (1 - ALPHA) * model_delta + ALPHA * mig_delta
    ok = np.isfinite(fused)
    new[i, pos4422[ok]] = fused[ok]

out = pd.DataFrame(new, index=sub.index, columns=test_cols)
out.index.name = 'sample_ID'
assert not out.isna().any().any() and np.isfinite(out.values).all()
out.to_csv(f'{DATA}/prediction_migfusion.csv')
print(f'[提交] prediction_migfusion.csv 生成 {out.shape}, alpha={ALPHA}')
