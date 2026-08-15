# -*- coding: utf-8 -*-
"""test_both 跨菌株迁移检验（用所有训练菌株的平均 Δ）"""
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
test_cols = traw.columns.tolist()
prot4422 = [l.strip() for l in open(f'{DATA}/prot_names.txt')]
col_of = {p: i for i, p in enumerate(test_cols)}
pos4422 = np.array([col_of[p] for p in prot4422], dtype=int)

def mk_key(df):
    return (df['data_source'].astype(str) + '|' + df['instrument'].astype(str) + '|'
            + df['Yeast_cell_plate'].astype(str) + '|' + df['Strains'].astype(str) + '|'
            + df['Medium'].astype(str) + '|' + df['Temperature'].astype(str) + '|'
            + df['pert_time'].astype(str)).values

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
drug_mu = {}
for c in set(chem_of[is_train]):
    m = np.isin(chem_of, c) & is_train
    drug_mu[c] = np.nanmean(delta_tr[m], axis=0)

pert2morgan = feats['pert2morgan64']
train_chems = [c for c in drug_mu if c in pert2morgan]

# test_both 的新化合物
idx = np.where(tmeta['split_final'].eq('test_both').values)[0]
treat_in = [p for p in idx if tmeta.iloc[p]['perturbation_no_concentration'] not in set(chem_of)]
t_log2_full = np.log2(traw.values.astype(np.float64))  # 全量 4454 行
t_log2 = np.log2(traw.iloc[treat_in].values.astype(np.float64))
t_chem = tmeta.iloc[treat_in]['perturbation_no_concentration'].values
t_key = mk_key(tmeta)
ctrl_te = {}
for k, pos in zip(t_key, np.where(tmeta['perturbation_no_concentration'].isin(['Water', 'DMSO']).values)[0]):
    ctrl_te.setdefault(k, []).append(pos)

print(f"{'化合物':<26}{'top1':<22}{'sim':>6}{'跨菌株PCC':>12}{'n':>6}")
for c in sorted(set(t_chem)):
    v = pert2morgan.get(c)
    if v is None:
        continue
    sims = []
    for tc in train_chems:
        w = pert2morgan[tc]
        s = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-8)
        sims.append((s, tc))
    sims.sort(reverse=True)
    tc1 = sims[0][1]
    samp = [i for i, cc in enumerate(t_chem) if cc == c]
    preds = []; truths = []
    for i in samp:
        # i 是过滤后索引，对应 treat_in[i] 的原始行
        rows = ctrl_te.get(t_key[treat_in[i]], [])
        if not rows:
            continue
        # 对照是全量 t_log2 的行（原始行号），需用全量 log2
        yc = np.nanmean(t_log2_full[rows], axis=0)
        dt = t_log2[i] - yc
        dt4422 = dt[pos4422]
        ok = np.isfinite(dt4422) & np.isfinite(drug_mu[tc1])
        truths.append(dt4422[ok]); preds.append(drug_mu[tc1][ok])
    if not preds:
        continue
    t_all = np.concatenate(truths); p_all = np.concatenate(preds)
    pcc = np.corrcoef(t_all, p_all)[0, 1] if len(t_all) > 50 else float('nan')
    print(f"{c:<26}{tc1:<22}{sims[0][0]:>6.3f}{pcc:>12.3f}{len(t_all):>6}")
