# -*- coding: utf-8 -*-
"""test 集六模块自评（官方已发布 test 真值，可自评不可训练）
两种对照口径：--ctrl test（test 内对照）/ --ctrl train（训练集对照池）
用法: python _test_score.py <提交csv> [--ctrl test|train]
"""
import numpy as np, pandas as pd, sys

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
INPUT = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"

args = sys.argv[1:]
SUBMIT = args[0] if args else f'{DATA}/prediction_final_5243.csv'
CTRL_MODE = 'test'
if '--ctrl' in args:
    CTRL_MODE = args[args.index('--ctrl') + 1]

meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float64)
mask = np.load(f'{DATA}/mask.npy').astype(bool)
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values
tr_y_nan = np.where(mask, y_log2, np.nan)

# ---------- test 数据 ----------
tmeta = pd.read_csv(f'{INPUT}/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')
traw = pd.read_csv(f'{INPUT}/WAYB_WAYC_proteome_raw_test.csv').set_index('sample_ID')
sub = pd.read_csv(SUBMIT, index_col=0)
assert (tmeta.index == sub.index).all() and (tmeta.index == traw.index).all()
cols = sub.columns.tolist()
t_log2 = np.log2(traw[cols].values.astype(np.float64))

# 蛋白列映射：训练 4422 -> 提交 5243
prot4422 = [l.strip() for l in open(f'{DATA}/prot_names.txt')]
col_of = {p: i for i, p in enumerate(cols)}
pos4422 = np.array([col_of[p] for p in prot4422], dtype=int)  # 5243 列中 4422 的位置

# 角色
is_ctrl = tmeta['perturbation_no_concentration'].isin(['Water', 'DMSO']).values
is_qc = tmeta['perturbation_no_concentration'].astype(str).str.contains('Quality', case=False, na=False).values
is_treat = (~is_ctrl) & (~is_qc)
treat_pos = np.where(is_treat)[0]
tpos_idx = {p: i for i, p in enumerate(treat_pos)}

def mk_key(df):
    return (df['data_source'].astype(str) + '|' + df['instrument'].astype(str) + '|'
            + df['Yeast_cell_plate'].astype(str) + '|' + df['Strains'].astype(str) + '|'
            + df['Medium'].astype(str) + '|' + df['Temperature'].astype(str) + '|'
            + df['pert_time'].astype(str)).values

# ---------- 对照 ----------
if CTRL_MODE == 'test':
    # test 内对照
    t_key = mk_key(tmeta)
    ctrl_lookup = {}
    for k, pos in zip(t_key, np.where(is_ctrl)[0]):
        ctrl_lookup.setdefault(k, []).append(pos)
    ctrl_vals = np.full((len(treat_pos), len(cols)), np.nan)
    for i, pos in enumerate(treat_pos):
        rows = ctrl_lookup.get(t_key[pos], [])
        if not rows:
            continue
        cvals = t_log2[rows]; cm = np.isfinite(cvals)
        with np.errstate(invalid='ignore'):
            ctrl_vals[i] = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, 0), 0) / cm.sum(0), np.nan)
else:
    # 训练集对照池 + test 对照池合并（server 口径：全局对照）
    ctrl_idx = np.where(meta['role'].eq('control').values)[0]
    ctrl_key = mk_key(meta.iloc[ctrl_idx])  # 只用对照行的 key！
    ctrl_lookup_tr = {}
    for k, pos in zip(ctrl_key, ctrl_idx):
        ctrl_lookup_tr.setdefault(k, []).append(pos)
    t_key = mk_key(tmeta)
    ctrl_lookup_te = {}
    for k, pos in zip(t_key, np.where(is_ctrl)[0]):
        ctrl_lookup_te.setdefault(k, []).append(pos)
    ctrl_vals = np.full((len(treat_pos), len(cols)), np.nan)
    for i, pos in enumerate(treat_pos):
        full = np.full(len(cols), np.nan)
        rows = ctrl_lookup_tr.get(t_key[pos], [])
        if rows:
            cvals = tr_y_nan[rows]
            cm = np.isfinite(cvals)
            with np.errstate(invalid='ignore'):
                m4422 = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, 0), 0) / cm.sum(0), np.nan)
            full[pos4422] = m4422
        rows = ctrl_lookup_te.get(t_key[pos], [])
        if rows:
            cvals = t_log2[rows]
            cm = np.isfinite(cvals)
            with np.errstate(invalid='ignore'):
                mfull = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, 0), 0) / cm.sum(0), np.nan)
            full = np.where(np.isfinite(full), full, mfull)
        ctrl_vals[i] = full

n_matched = np.isfinite(ctrl_vals).any(axis=1).sum()
print(f'[对照] 模式={CTRL_MODE} | 处理样本 {len(treat_pos)}，有对照值的样本 {n_matched}')

# ---------- 训练集 μ_ctx / μ_drug（仅训练行）----------
treat_all = np.where(meta['role'].eq('treatment').values)[0]
train_treat = np.where(train_mask & meta['role'].eq('treatment').values)[0]
is_train_treat = np.isin(treat_all, train_treat)
tr_treat_idx = np.where(is_train_treat)[0]

# 训练集 Δ（用训练集对照）
ctrl_idx_tr = np.where(meta['role'].eq('control').values)[0]
ctrl_key_tr = mk_key(meta.iloc[ctrl_idx_tr])  # 只用对照行！
ctrl_lookup_tr = {}
for k, pos in zip(ctrl_key_tr, ctrl_idx_tr):
    ctrl_lookup_tr.setdefault(k, []).append(pos)
ctrl_tr = np.full((len(treat_all), P), np.nan)
for i, sid in enumerate(treat_all):
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    rows = ctrl_lookup_tr.get(k, [])
    if not rows:
        continue
    cvals = tr_y_nan[rows]; cm = mask[rows]
    with np.errstate(invalid='ignore'):
        ctrl_tr[i] = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)
delta_tr = tr_y_nan[treat_all] - ctrl_tr

chem_of = meta['perturbation_no_concentration'].values[treat_all]
ctx_key = (meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|'
           + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)).values[treat_all]

mu_ctx_pool = {}; mu_drug_pool = {}
for key, members in pd.Series(tr_treat_idx, index=ctx_key[is_train_treat]).groupby(level=0):
    mu_ctx_pool[key] = np.nanmean(delta_tr[members.values], axis=0)
for key, members in pd.Series(tr_treat_idx, index=chem_of[is_train_treat]).groupby(level=0):
    mu_drug_pool[key] = np.nanmean(delta_tr[members.values], axis=0)
def expand(m4422):
    full = np.zeros(len(cols))
    full[pos4422] = m4422
    return full
mu_ctx_pool = {k: expand(v) for k, v in mu_ctx_pool.items()}
mu_drug_pool = {k: expand(v) for k, v in mu_drug_pool.items()}

t_ctx = (tmeta['Strains'].astype(str) + '|' + tmeta['Medium'].astype(str) + '|'
         + tmeta['Temperature'].astype(str) + '|' + tmeta['pert_time'].astype(str)).values
t_chem = tmeta['perturbation_no_concentration'].values

# ---------- 指标 ----------
def safe_pcc(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10: return 0.0
    a, b = a[ok], b[ok]
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0

def safe_r2(yp, yt, ok):
    if ok.sum() < 3: return 0.0
    a, b = yp[ok], yt[ok]
    ss_res = ((a - b) ** 2).sum(); ss_tot = ((b - b.mean()) ** 2).sum()
    if ss_tot < 1e-12: return 0.0
    r2 = 1 - ss_res / ss_tot
    return float(r2) if np.isfinite(r2) else 0.0

print(f"\n=== 提交: {SUBMIT.split(chr(92))[-1].split('/')[-1]} | 对照口径={CTRL_MODE} ===")
scenes = ['test_chem_only', 'test_strain_only', 'test_both', 'test_time']
print(f"{'场景':<16}{'M1:FC(25%)':>10}{'M2:绝对(20%)':>12}{'M3:ctx残差(20%)':>15}{'M4:drug残差(20%)':>15}{'M5:双盲(10%)':>12}{'M6:DEP(5%)':>11}{'总分':>8}")
print('-' * 115)

scores = {}
for scene in scenes:
    idx = np.where(tmeta['split_final'].eq(scene).values)[0]
    treat_in_scene = [p for p in idx if is_treat[p]]
    if not treat_in_scene:
        continue
    pos_arr = np.array(treat_in_scene)
    yt = t_log2[pos_arr]
    yp = sub.iloc[pos_arr].values.astype(np.float64)
    mt = np.isfinite(yt)
    yc = ctrl_vals[[tpos_idx[p] for p in pos_arr]]
    dt = yt - yc
    dp = yp - yc
    fc_ok = np.isfinite(dp) & np.isfinite(dt) & mt
    mod1 = safe_pcc(dp[fc_ok].ravel(), dt[fc_ok].ravel())

    p_r2s = []
    for pp in range(yp.shape[1]):
        ok = mt[:, pp] & np.isfinite(yp[:, pp])
        if ok.sum() >= 3:
            p_r2s.append(safe_r2(yp[:, pp], yt[:, pp], ok))
    prot_r2 = float(np.median(p_r2s)) if p_r2s else 0.0
    s_r2s = []
    for s in range(len(pos_arr)):
        ok = mt[s] & np.isfinite(yp[s])
        if ok.sum() >= 10:
            s_r2s.append(safe_r2(yp[s], yt[s], ok))
    samp_r2 = float(np.median(s_r2s)) if s_r2s else 0.0
    mod2 = 0.5 * max(prot_r2, 0) + 0.5 * max(samp_r2, 0)

    mu_c = np.array([mu_ctx_pool.get(t_ctx[p], np.zeros(len(cols))) for p in pos_arr])
    mu_d = np.array([mu_drug_pool.get(t_chem[p], np.zeros(len(cols))) for p in pos_arr])
    mod3 = safe_pcc((dp - mu_c)[fc_ok].ravel(), (dt - mu_c)[fc_ok].ravel())
    mod4 = safe_pcc((dp - mu_d)[fc_ok].ravel(), (dt - mu_d)[fc_ok].ravel())

    if scene == 'test_both':
        mod5 = 0.5 * mod1 + 0.5 * prot_r2
    elif scene == 'test_time':
        mod5 = 0.7 * prot_r2 + 0.3 * mod1
    else:
        mod5 = None
    mod5 = max(mod5, 0) if mod5 is not None else None

    hi = np.abs(dt) > 1
    dep_acc = 0.0; dep_pcc = 0.0; dep_f1k = 0.0
    if hi.sum() > 0:
        dp_hi = dp[hi & fc_ok]; dt_hi = dt[hi & fc_ok]
        if len(dp_hi) > 10:
            dep_acc = float((np.sign(dp_hi) == np.sign(dt_hi)).mean())
            dep_pcc = safe_pcc(dp_hi.ravel(), dt_hi.ravel())
            # F1@K：逐样本按 |Δ_pred| 排序取固定 top-K（50/100/200 三档均值），
            # precision/recall 分离，避免「全报阳性」刷高召回（官方规则明确要求）
            f1s = []
            for K in (50, 100, 200):
                f1k = []
                for s in range(len(pos_arr)):
                    ok = fc_ok[s]
                    if ok.sum() < 5:
                        continue
                    dp_s = dp[s][ok]; dt_s = dt[s][ok]
                    hi_s = np.abs(dt_s) > 1
                    if hi_s.sum() == 0:
                        continue
                    order = np.argsort(-np.abs(dp_s))
                    pred_hi = np.zeros(len(dp_s), dtype=bool)
                    pred_hi[order[:min(K, len(dp_s))]] = True
                    tp = (pred_hi & hi_s).sum()
                    prec = tp / max(pred_hi.sum(), 1)
                    rec = tp / hi_s.sum()
                    f1k.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
                if f1k:
                    f1s.append(float(np.mean(f1k)))
            dep_f1k = float(np.mean(f1s)) if f1s else 0.0
    mod6 = 0.4 * dep_acc + 0.3 * max(dep_pcc, 0) + 0.3 * dep_f1k

    total = 0.25*mod1 + 0.20*mod2 + 0.20*mod3 + 0.20*mod4
    if mod5 is not None:
        total += 0.10*mod5
    total += 0.05*mod6
    scores[scene] = (mod1, mod2, mod3, mod4, mod5, mod6, total, len(pos_arr))
    m5s = f"{mod5:.3f}" if mod5 is not None else "N/A"
    print(f"{scene:<16}{mod1:>10.3f}{mod2:>12.3f}{mod3:>15.3f}{mod4:>15.3f}{m5s:>12}{mod6:>11.3f}{total:>8.3f}  n={len(pos_arr)}")

print()
tw = 0; ws = 0
for scene, w in [('test_chem_only', 1), ('test_strain_only', 1), ('test_both', 0.5), ('test_time', 0.5)]:
    if scene in scores:
        tw += scores[scene][-2] * w; ws += w
print(f"加权总分 = {tw/ws:.4f}")
