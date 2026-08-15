# -*- coding: utf-8 -*-
"""正确的化学迁移检验：条件对齐 + top-k 加权
test_chem_only = 新化合物 + 已见菌株(CEK/DHY210/CGD/BAH)
正确做法：对每个 test 样本，找指纹最相似的训练化合物，
用训练化合物在【相同菌株/培养基/温度/时间】下的 Δ 做预测。
这才是化学结构信息能否迁移的公平检验。
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

def mk_ctx_key(df):
    return df['Strains'].astype(str).to_numpy()  # 只对齐菌株（test_chem_only 本质：已见菌株+新化合物）

# 训练集 matched control
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

# 训练集（仅 train 划分）样本索引
train_mask = meta['split_final'].eq('train').values
is_train = np.isin(treat_all, np.where(train_mask & meta['role'].eq('treatment').values)[0])
tr_ctx = mk_ctx_key(meta)[treat_all]
tr_chem = meta['perturbation_no_concentration'].values[treat_all]

# 按 (化合物, ctx) 分组取 Δ 均值（训练内）
drug_ctx_mu = {}
for i in np.where(is_train)[0]:
    key = (tr_chem[i], tr_ctx[i])
    if key not in drug_ctx_mu:
        drug_ctx_mu[key] = []
    drug_ctx_mu[key].append(delta_tr[i])
for k in drug_ctx_mu:
    drug_ctx_mu[k] = np.nanmean(np.stack(drug_ctx_mu[k]), axis=0)

# 指纹相似度
pert2morgan = feats['pert2morgan64'] if 'pert2morgan64' in feats else None
if pert2morgan is None:
    morgan64 = feats['chem_morgan']
    pert2morgan = {}
    for i, p in enumerate(meta['perturbation_no_concentration'].values):
        pert2morgan.setdefault(p, morgan64[i])

train_chems = sorted(set(tr_chem[is_train]))
print(f'训练化合物 {len(train_chems)} 个, 有指纹的 {sum(1 for c in train_chems if c in pert2morgan)} 个')

def topk_train_chems(test_c, k=5):
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

# test_chem_only 样本
idx = np.where(tmeta['split_final'].eq('test_chem_only').values)[0]
treat_in = [p for p in idx if not tmeta.iloc[p]['perturbation_no_concentration'] in ('Water', 'DMSO')]
t_log2 = np.log2(traw.iloc[treat_in].values.astype(np.float64))
t_ctx_arr = mk_ctx_key(tmeta)[treat_in]
t_chem_arr = tmeta.iloc[treat_in]['perturbation_no_concentration'].values

# 用训练集对照算 test Δ（server 口径：训练对照池）
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

print(f"\n{'化合物':<26}{'top1':<22}{'sim1':>6}{'top2':<22}{'sim2':>6}{'迁移PCC':>9}{'n':>6}")
total_ok = 0; total_pcc = []
for c in sorted(set(t_chem_arr)):
    tops = topk_train_chems(c, k=2)
    if not tops:
        continue
    samp = [i for i, cc in enumerate(t_chem_arr) if cc == c]
    # top-k 加权迁移：同菌株下，按 sim 加权的训练化合物 Δ
    preds = []; truths = []
    for i in samp:
        ctx = t_ctx_arr[i]
        dt = test_delta[i]
        dt4422 = dt[pos4422]
        # 收集同菌株可用的 top-k 训练化合物
        mus = []; ws = []
        for sim, tc in tops:
            mu = drug_ctx_mu.get((tc, ctx))
            if mu is None:
                continue
            mus.append(mu); ws.append(max(sim, 0.0))
        if not mus:
            continue
        ws = np.array(ws); ws = ws / ws.sum()
        mu_w = np.sum(np.stack([w * m for w, m in zip(ws, mus)]), axis=0)
        ok = np.isfinite(dt4422) & np.isfinite(mu_w)
        truths.append(dt4422[ok]); preds.append(mu_w[ok])
    if not preds:
        print(f"{c:<26}{tops[0][1]:<22}{tops[0][0]:>6.3f}{tops[1][1] if len(tops)>1 else '':<22}{(tops[1][0] if len(tops)>1 else 0):>6.3f}{'nan':>9}{0:>6}")
        continue
    t_all = np.concatenate(truths); p_all = np.concatenate(preds)
    if len(t_all) < 50:
        print(f"{c:<26}{tops[0][1]:<22}{tops[0][0]:>6.3f}{tops[1][1] if len(tops)>1 else '':<22}{(tops[1][0] if len(tops)>1 else 0):>6.3f}{float('nan'):>9}{len(t_all):>6}")
        continue
    pcc = np.corrcoef(t_all, p_all)[0, 1]
    total_ok += 1
    total_pcc.append(pcc)
    print(f"{c:<26}{tops[0][1]:<22}{tops[0][0]:>6.3f}{tops[1][1] if len(tops)>1 else '':<22}{(tops[1][0] if len(tops)>1 else 0):>6.3f}{pcc:>9.3f}{len(t_all):>6}")

if total_pcc:
    print(f"\n平均迁移 PCC (n={total_ok} 化合物): {np.mean(total_pcc):.3f}")
