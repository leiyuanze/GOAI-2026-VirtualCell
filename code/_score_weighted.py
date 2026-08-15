# -*- coding: utf-8 -*-
"""v35 权重变体在 test 上的六模块评估（复用 _test_score 逻辑，扩展为任意权重组合）
用法: python _score_weighted.py <输出csv> <模型列表以,分隔> <权重以,分隔>
"""
import numpy as np, pandas as pd, torch, importlib.util, pickle, sys, hashlib

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
BASE = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# 参数: <out_csv> <models> <weights>
out_csv = sys.argv[1] if len(sys.argv) > 1 else 'weighted.csv'
model_names = sys.argv[2].split(',') if len(sys.argv) > 2 else ['v21a', 'v21b', 'v21c', 'v35', 'v35']
weights = [float(w) for w in sys.argv[3].split(',')] if len(sys.argv) > 3 else [1, 1, 1, 2]

meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float64)
mask = np.load(f'{DATA}/mask.npy').astype(bool)
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values
tr_y_nan = np.where(mask, y_log2, np.nan)
ctx_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)

def load(p, cls, sa=False):
    m = cls(feats, P=P)
    m.load_state_dict(torch.load(f'{DATA}/{p}', map_location=DEV, weights_only=True))
    if sa:
        m.set_strain_avg()
    return m.to(DEV).eval()

_s21 = importlib.util.spec_from_file_location('m21', r'D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v21.py')
_m21 = importlib.util.module_from_spec(_s21); _s21.loader.exec_module(_m21)
_s30 = importlib.util.spec_from_file_location('m30', r'D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v30.py')
_m30 = importlib.util.module_from_spec(_s30); _s30.loader.exec_module(_m30)
_s37 = importlib.util.spec_from_file_location('m37', r'D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v37.py')
_m37 = importlib.util.module_from_spec(_s37); _s37.loader.exec_module(_m37)
_s49 = importlib.util.spec_from_file_location('m49', r'D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v49.py')
_m49 = importlib.util.module_from_spec(_s49); _s49.loader.exec_module(_m49)

MODELS = {
    'v21a': load('model_v21.pt', _m21.VCellModel),
    'v21b': load('model_v21_s43.pt', _m21.VCellModel),
    'v21c': load('model_v21_s44.pt', _m21.VCellModel),
    'v30': load('model_v30_best.pt', _m30.VCellModel, True),
    'v35': load('model_v35_best.pt', _m30.VCellModel, True),
    'v37': load('model_v37_42_best.pt', _m37.VCellModel, True),
    'v49': load('model_v49_42_best.pt', _m49.VCellModel, True),
}
print('[模型] 加载完成', flush=True)

# ---------- test 特征 ----------
tmeta = pd.read_csv(f'{BASE}/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')
strains_tr = sorted(meta['Strains'].unique()); chems_tr = sorted(meta.loc[meta['role'].eq('treatment'), 'perturbation_no_concentration'].unique())
strain2id = {s: i for i, s in enumerate(strains_tr)}; chem2id = {c: i for i, c in enumerate(chems_tr)}
sm_cats = sorted((meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str)).unique())
ct_cats = sorted((meta['perturbation_no_concentration'].astype(str) + '|' + meta['Temperature'].astype(str)).unique())
sm2id = {k: i for i, k in enumerate(sm_cats)}; ct2id = {k: i for i, k in enumerate(ct_cats)}
src_cats = sorted(meta['data_source'].unique()); ins_cats = sorted(meta['instrument'].unique()); plt_cats = sorted(meta['Yeast_cell_plate'].unique())
src2id = {k: i for i, k in enumerate(src_cats)}; ins2id = {k: i for i, k in enumerate(ins_cats)}; plt2id = {k: i for i, k in enumerate(plt_cats)}
train_strains = set(meta.loc[meta['split_final'].eq('train'), 'Strains'])
train_chems = set(meta.loc[meta['split_final'].eq('train') & meta['role'].eq('treatment'), 'perturbation_no_concentration'])

def hash_vec(name, dim=32):
    h = hashlib.sha256(str(name).encode()).hexdigest()
    return np.array([int(h[i*2:i*2+2], 16) / 255.0 for i in range(dim)])

tstrain = np.array([strain2id.get(s, -1) for s in tmeta['Strains']], dtype=np.int64)
tchem = np.array([chem2id.get(c, -1) for c in tmeta['perturbation_no_concentration']], dtype=np.int64)
tmed = np.array([[1.0 if m == mm else 0.0 for mm in sorted(meta['Medium'].unique())] for m in tmeta['Medium']], dtype=np.float32)
ttemp = ((tmeta['Temperature'].astype(float) - 30.0) / 7.0).values.astype(np.float32)
tt = tmeta['pert_time'].astype(float).values; tt_log = np.log2(tt / 15.0) / np.log2(240.0 / 15.0)
ttfeat = np.stack([tt_log, np.sin(2*np.pi*tt_log), np.cos(2*np.pi*tt_log)], axis=1).astype(np.float32)
tsm = np.array([sm2id.get(f"{s}|{m}", -1) for s, m in zip(tmeta['Strains'], tmeta['Medium'])], dtype=np.int64)
tct = np.array([ct2id.get(f"{c}|{t_}", -1) for c, t_ in zip(tmeta['perturbation_no_concentration'], tmeta['Temperature'])], dtype=np.int64)
tsrc = np.array([src2id.get(s, -1) for s in tmeta['data_source']], dtype=np.int64)
tins = np.array([ins2id.get(s, -1) for s in tmeta['instrument']], dtype=np.int64)
tplt = np.array([plt2id.get(s, -1) for s in tmeta['Yeast_cell_plate']], dtype=np.int64)
tchash = np.array([hash_vec(c) for c in tmeta['perturbation_no_concentration']], dtype=np.float32)
tcseen = np.array([1.0 if c in train_chems else 0.0 for c in tmeta['perturbation_no_concentration']], dtype=np.float32)
tsseen = np.array([1.0 if s in train_strains else 0.0 for s in tmeta['Strains']], dtype=np.float32)
pert2morgan = {}
for i, p in enumerate(meta['perturbation_no_concentration'].values):
    pert2morgan.setdefault(p, feats['chem_morgan'][i])
tmorgan = np.array([pert2morgan.get(c, np.zeros(64, dtype=np.float32)) for c in tmeta['perturbation_no_concentration']], dtype=np.float32)
gmean = feats['gmean']
ctx_key_tr = (meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|'
              + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)).values
t_ctx_prior = np.tile(gmean, (len(tmeta), 1)).astype(np.float32)
ctx_grp = {}
for i in np.where(train_mask)[0]:
    ctx_grp.setdefault(ctx_key_tr[i], []).append(tr_y_nan[i])
for k, vals in ctx_grp.items():
    ctx_grp[k] = np.nanmean(vals, axis=0)
t_ctx = (tmeta['Strains'].astype(str) + '|' + tmeta['Medium'].astype(str) + '|'
         + tmeta['Temperature'].astype(str) + '|' + tmeta['pert_time'].astype(str)).values
for i, k in enumerate(t_ctx):
    if k in ctx_grp:
        t_ctx_prior[i] = ctx_grp[k]
t_ctx_prior = np.nan_to_num(t_ctx_prior, nan=0.0)

def tpred_x(names, ws):
    preds = []
    with torch.no_grad():
        for n, w in zip(names, ws):
            x = {'bio': [torch.from_numpy(tstrain), torch.from_numpy(tchem), torch.from_numpy(tchash),
                         torch.from_numpy(tmed), torch.from_numpy(ttemp), torch.from_numpy(ttfeat),
                         torch.from_numpy(tsm), torch.from_numpy(tct)],
                 'ctx': [torch.from_numpy(tsrc), torch.from_numpy(tins), torch.from_numpy(tplt)],
                 'seen': [torch.from_numpy(tcseen), torch.from_numpy(tsseen)]}
            if n in ('v30', 'v35', 'v37', 'v49'):
                x['ctx_prior'] = torch.from_numpy(t_ctx_prior)
                x['chem_morgan'] = torch.from_numpy(tmorgan)
            xg = {k: (v.to(DEV) if k in ('ctx_prior', 'chem_morgan') else [t.to(DEV) for t in v]) for k, v in x.items()}
            preds.append(w * MODELS[n](xg).cpu().numpy())
    return np.sum(preds, axis=0) / sum(ws)

# ---------- 评估（复用 _test_score 逻辑）----------
traw = pd.read_csv(f'{BASE}/WAYB_WAYC_proteome_raw_test.csv').set_index('sample_ID')
test_cols = traw.columns.tolist()
col_of = {p: i for i, p in enumerate(test_cols)}
prot4422 = [l.strip() for l in open(f'{DATA}/prot_names.txt')]
pos4422 = np.array([col_of[p] for p in prot4422 if p in col_of], dtype=int)
t_log2 = np.log2(traw[test_cols].values.astype(np.float64))
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

# 训练对照池 + test 对照池
ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = mk_key(meta.iloc[ctrl_idx])
ctrl_lookup_tr = {}
for k, pos in zip(ctrl_key, ctrl_idx):
    ctrl_lookup_tr.setdefault(k, []).append(pos)
t_key = mk_key(tmeta)
ctrl_lookup_te = {}
for k, pos in zip(t_key, np.where(is_ctrl)[0]):
    ctrl_lookup_te.setdefault(k, []).append(pos)
ctrl_vals = np.full((len(treat_pos), len(test_cols)), np.nan)
for i, pos in enumerate(treat_pos):
    full = np.full(len(test_cols), np.nan)
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

# μ_ctx / μ_drug（训练 LOO pool）
treat_all = np.where(meta['role'].eq('treatment').values)[0]
train_treat = np.where(train_mask & meta['role'].eq('treatment').values)[0]
is_train_treat = np.isin(treat_all, train_treat)
tr_treat_idx = np.where(is_train_treat)[0]
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
ctx_key2 = (meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|'
            + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)).values[treat_all]
mu_ctx_pool = {}; mu_drug_pool = {}
for key, members in pd.Series(tr_treat_idx, index=ctx_key2[is_train_treat]).groupby(level=0):
    mu_ctx_pool[key] = np.nanmean(delta_tr[members.values], axis=0)
for key, members in pd.Series(tr_treat_idx, index=chem_of[is_train_treat]).groupby(level=0):
    mu_drug_pool[key] = np.nanmean(delta_tr[members.values], axis=0)
def expand(m4422):
    full = np.zeros(len(test_cols))
    full[pos4422] = m4422
    return full
mu_ctx_pool = {k: expand(v) for k, v in mu_ctx_pool.items()}
mu_drug_pool = {k: expand(v) for k, v in mu_drug_pool.items()}

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

# 训练 log2 均值（5243 填充）
meta_full = pd.read_csv(f'{BASE}/WAYB_WAYC_metadata_train_val(1).csv')
prot_full = pd.read_csv(f'{BASE}/WAYB_WAYC_proteome_raw_train_val.csv', index_col='sample_ID')
train_ids = meta_full.loc[meta_full['split_final'].eq('train'), 'sample_ID']
lm = np.log2(prot_full.loc[train_ids]).mean(axis=0, skipna=True)
lm = lm.fillna(np.median(lm.dropna()))

pred4422 = tpred_x(model_names, weights)
print(f"[预测] 完成, 形状 {pred4422.shape}", flush=True)

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
    p4422 = pred4422[pos_arr]
    yp = np.full((len(pos_arr), len(test_cols)), np.nan, dtype=np.float64)
    yp[:, pos4422] = p4422
    for c in range(len(test_cols)):
        mcol = np.isnan(yp[:, c])
        if mcol.any():
            yp[mcol, c] = lm[test_cols[c]]
    yt = t_log2[pos_arr]
    mt = np.isfinite(yt)
    yc = ctrl_vals[[tpos_idx[p] for p in pos_arr]]
    dt = yt - yc; dp = yp - yc
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
    mu_c = np.array([mu_ctx_pool.get(t_ctx[p], np.zeros(len(test_cols))) for p in pos_arr])
    mu_d = np.array([mu_drug_pool.get(tmeta['perturbation_no_concentration'].values[p], np.zeros(len(test_cols))) for p in pos_arr])
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
    dep_acc = 0.0; dep_pcc = 0.0
    if hi.sum() > 0:
        dp_hi = dp[hi & fc_ok]; dt_hi = dt[hi & fc_ok]
        if len(dp_hi) > 10:
            dep_acc = float((np.sign(dp_hi) == np.sign(dt_hi)).mean())
            dep_pcc = safe_pcc(dp_hi.ravel(), dt_hi.ravel())
    mod6 = 0.5 * dep_acc + 0.5 * max(dep_pcc, 0)
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

# 输出 5243 提交
sub = pd.DataFrame(np.full((len(tmeta), len(test_cols)), np.nan), index=tmeta.index, columns=test_cols)
sub.iloc[:, pos4422] = pred4422
for c in range(len(test_cols)):
    mcol = sub.iloc[:, c].isna()
    if mcol.any():
        sub.iloc[mcol, c] = lm[test_cols[c]]
sub.index.name = 'sample_ID'
sub.to_csv(f'{DATA}/{out_csv}')
print(f"[提交] {out_csv} 生成 {sub.shape}")
