# -*- coding: utf-8 -*-
"""迁移融合 v4（自适应 α，gpt2 步骤10/P1-3，8/16）
α 由 val 伪新化合物验证（_mig_val_sweep.py）确定的形式：
  α(sim) = 0.1 + 0.2 × clamp((sim - 0.1) / 0.4, 0, 1)
  - 相似度 < 0.1（几乎无相似）：α=0.1（FCCP 验证：低相似必须小 α 防噪声）
  - 相似度 ≥ 0.5：α=0.3（Raloxifene 验证：中相似略大 α 更好）
★ 合规：迁移池/对照池仅 train 划分
用法：python 08c_mig_fusion_adaptive.py <base_submit.csv> <out.csv>
"""
import sys, numpy as np, pandas as pd, pickle

DATA = 'data'
BASE_SUBMIT = sys.argv[1] if len(sys.argv) > 1 else f'{DATA}/prediction_v50ens_base.csv'
OUT = sys.argv[2] if len(sys.argv) > 2 else f'{DATA}/prediction_v50ens_adaptive.csv'

meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float64)
mask = np.load(f'{DATA}/mask.npy').astype(bool)
tr_y_nan = np.where(mask, y_log2, np.nan)
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))
tmeta = pd.read_csv('../input/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')
sub = pd.read_csv(BASE_SUBMIT, index_col=0)
P = y_log2.shape[1]
test_cols = sub.columns.tolist()
prot4422 = [l.strip() for l in open(f'{DATA}/prot_names.txt')]
col_of = {p: i for i, p in enumerate(test_cols)}
pos4422 = np.array([col_of[p] for p in prot4422], dtype=int)
pert2morgan = feats['pert2morgan64']
train_mask = meta['split_final'].eq('train').values

def mk_key(df):
    return (df['data_source'].astype(str) + '|' + df['instrument'].astype(str) + '|'
            + df['Yeast_cell_plate'].astype(str) + '|' + df['Strains'].astype(str) + '|'
            + df['Medium'].astype(str) + '|' + df['Temperature'].astype(str) + '|'
            + df['pert_time'].astype(str)).values

# 训练对照池（仅 train）
ctrl_idx = np.where(meta['role'].eq('control').values & train_mask)[0]
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
is_train = np.isin(treat_all, np.where(train_mask & meta['role'].eq('treatment').values)[0])
tr_strain = meta['Strains'].values[treat_all]

drug_strain_mu = {}
for i in np.where(is_train)[0]:
    key = (tr_strain[i], chem_of[i])
    drug_strain_mu.setdefault(key, []).append(delta_tr[i])
for k in drug_strain_mu:
    drug_strain_mu[k] = np.nanmean(np.stack(drug_strain_mu[k]), axis=0)
# ★ gpt2 步骤10 迁移优先级：同菌株 → 同培养基 → 全局（回退池）
tr_med = meta['Medium'].values[treat_all]
drug_med_mu = {}
for i in np.where(is_train)[0]:
    key = (tr_med[i], chem_of[i])
    drug_med_mu.setdefault(key, []).append(delta_tr[i])
for k in drug_med_mu:
    drug_med_mu[k] = np.nanmean(np.stack(drug_med_mu[k]), axis=0)
drug_global_mu = {}
for i in np.where(is_train)[0]:
    drug_global_mu.setdefault(chem_of[i], []).append(delta_tr[i])
for k in drug_global_mu:
    drug_global_mu[k] = np.nanmean(np.stack(drug_global_mu[k]), axis=0)

# ★ 迁移 2.0（gpt3 §五.3 / opus3 路线2）：上下文均值池（菌株|培养基|温度|时间 → Δ 均值）
tr_ctx = (meta['Strains'].astype(str).values[treat_all] + '|' + meta['Medium'].astype(str).values[treat_all] + '|'
          + meta['Temperature'].astype(str).values[treat_all] + '|' + meta['pert_time'].astype(str).values[treat_all])
ctx_mu_pool = {}
for i in np.where(is_train)[0]:
    ctx_mu_pool.setdefault(tr_ctx[i], []).append(delta_tr[i])
for k in ctx_mu_pool:
    ctx_mu_pool[k] = np.nanmean(np.stack(ctx_mu_pool[k]), axis=0)
global_delta_mu = np.nanmean(np.stack([delta_tr[i] for i in np.where(is_train)[0]]), axis=0)

def ctx_mu_of(strain, med, temp, time):
    """上下文处理均值（train-only）；无匹配回退全局"""
    return ctx_mu_pool.get(f'{strain}|{med}|{temp}|{time}', global_delta_mu)

def mig_mu_of(strain, med, tc):
    """迁移 μ 回退链：同菌株 → 同培养基 → 全局（gpt2 步骤10 优先级 3/4 级）"""
    mu = drug_strain_mu.get((strain, tc))
    if mu is not None:
        return mu
    mu = drug_med_mu.get((med, tc))
    if mu is not None:
        return mu
    return drug_global_mu.get(tc)

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
        sims.append((float(s), tc))
    sims.sort(reverse=True)
    return sims[:k]

def alpha_of(sim):
    """★ 迁移 2.0 α（gpt3 §五.4 / opus1 5.1）：α=0.4·clip((sim−0.35)/0.65)，sim<0.35 不迁移"""
    if sim < 0.35:
        return 0.0
    return 0.4 * min(max((sim - 0.35) / 0.65, 0.0), 1.0)

def matched_control_mean(rows_src, src):
    cvals = src[rows_src]; cm = np.isfinite(cvals)
    with np.errstate(invalid='ignore'):
        return np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, 0), 0) / cm.sum(0), np.nan)

new = sub.copy().values.astype(np.float64)
is_new_chem = tmeta['perturbation_no_concentration'].isin(
    [c for c in tmeta['perturbation_no_concentration'].unique() if c not in set(chem_of)]
).values

n_fused = 0
for i in range(len(tmeta)):
    if not is_new_chem[i]:
        continue
    r = tmeta.iloc[i]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    rows = ctrl_lookup.get(k, [])
    if not rows:
        continue
    yc = matched_control_mean(rows, tr_y_nan)
    if not np.isfinite(yc).any():
        continue
    c = r['perturbation_no_concentration']
    tops = topk_sim(c)
    if not tops:
        continue
    alpha = alpha_of(tops[0][0])  # ★ 迁移 2.0：相似度门槛（S_max<0.35 不迁移）
    if alpha <= 0:
        continue
    TAU = 0.1  # ★ gpt2 步骤10：exp(sim/τ) softmax 权重（τ∈{0.05,0.1,0.2}，取 0.1）
    mus = []; ctxs = []; ws = []
    for sim, tc in tops:
        mu = mig_mu_of(r['Strains'], r['Medium'], tc)  # ★ gpt2 步骤10 回退链
        if mu is None:
            continue
        mus.append(mu)
        ctxs.append(ctx_mu_of(r['Strains'], r['Medium'], r['Temperature'], r['pert_time']))
        ws.append(np.exp(sim / TAU))
    if not mus:
        continue
    ws = np.array(ws); ws = ws / ws.sum()
    # ★ 迁移 2.0（gpt3 §五.3 / opus3 路线2）：迁移"特异响应" Δ−μ_context，融合时加回 μ_context
    mig_specific = np.sum(np.stack([w * (m - cmu) for w, m, cmu in zip(ws, mus, ctxs)]), axis=0)
    mig_specific = np.where(np.isfinite(mig_specific), mig_specific, 0.0)
    mu_ctx_test = ctx_mu_of(r['Strains'], r['Medium'], r['Temperature'], r['pert_time'])
    model_delta = new[i, pos4422] - yc
    fused = yc + (1 - alpha) * model_delta + alpha * (mu_ctx_test + mig_specific)
    ok = np.isfinite(fused)
    new[i, pos4422[ok]] = fused[ok]
    n_fused += 1

out = pd.DataFrame(new, index=sub.index, columns=test_cols)
out.index.name = 'sample_ID'
assert not out.isna().any().any() and np.isfinite(out.values).all()
out.to_csv(OUT)
print(f'[提交] {OUT} 生成 {out.shape} | 融合样本 {n_fused} | α=0.4·clip((sim−0.35)/0.65), sim<0.35 不迁移')
