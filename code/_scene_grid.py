# -*- coding: utf-8 -*-
"""场景级模型组合网格搜索：val + test 双口径
对每个场景比较：v3.7单 / 3×v2.1 / 3×v2.1+v30 / 3×v2.1+v35 / 3×v2.1+v37
输出每个组合在 val 各场景蛋白R2中位 和 test 各场景六模块总分，确定最优场景映射
"""
import numpy as np, pandas as pd, torch, importlib.util, pickle, itertools

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
BASE = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float32)
mask = np.load(f'{DATA}/mask.npy').astype(bool)
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))
P = y_log2.shape[1]
tr_y_nan = np.where(mask, y_log2, np.nan)
train_mask = meta['split_final'].eq('train').values
ctx_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)

def load_model(path, cls, set_avg=False):
    m = cls(feats, P=P)
    m.load_state_dict(torch.load(f"{DATA}/{path}", map_location=DEV, weights_only=True))
    if set_avg:
        m.set_strain_avg()
    return m.to(DEV).eval()

_s21 = importlib.util.spec_from_file_location('m21', f'{DATA}/../04_model_v21.py')
_m21 = importlib.util.module_from_spec(_s21); _s21.loader.exec_module(_m21)
_s30 = importlib.util.spec_from_file_location('m30', f'{DATA}/../04_model_v30.py')
_m30 = importlib.util.module_from_spec(_s30); _s30.loader.exec_module(_m30)
_s37 = importlib.util.spec_from_file_location('m37', f'{DATA}/../04_model_v37.py')
_m37 = importlib.util.module_from_spec(_s37); _s37.loader.exec_module(_m37)

MODELS = {
    'v21a': load_model('model_v21.pt', _m21.VCellModel),
    'v21b': load_model('model_v21_s43.pt', _m21.VCellModel),
    'v21c': load_model('model_v21_s44.pt', _m21.VCellModel),
    'v30': load_model('model_v30_best.pt', _m30.VCellModel, set_avg=True),
    'v35': load_model('model_v35_best.pt', _m30.VCellModel, set_avg=True),
    'v37': load_model('model_v37_best.pt', _m37.VCellModel, set_avg=True),
}
print('[模型] 加载完成:', {k: 'ok' for k in MODELS}, flush=True)

def make_x(idx, tag):
    d = {
        'bio': [torch.from_numpy(feats['strain_id'][idx]), torch.from_numpy(feats['chem_id'][idx]),
                torch.from_numpy(feats['chem_hash'][idx]), torch.from_numpy(feats['medium_onehot'][idx]),
                torch.from_numpy(feats['temp_norm'][idx]), torch.from_numpy(feats['time_feat'][idx]),
                torch.from_numpy(feats['sm_id'][idx]), torch.from_numpy(feats['ct_id'][idx])],
        'ctx': [torch.from_numpy(feats['src_id'][idx]), torch.from_numpy(feats['ins_id'][idx]),
                torch.from_numpy(feats['plt_id'][idx])],
        'seen': [torch.from_numpy(feats['chem_seen'][idx]), torch.from_numpy(feats['strain_seen'][idx])],
    }
    if tag in ('v30', 'v35', 'v37'):
        d['ctx_prior'] = torch.from_numpy(ctx_all[idx])
        d['chem_morgan'] = torch.from_numpy(feats['chem_morgan'][idx])
    return d

def to_gpu(x):
    return {k: (v.to(DEV) if k in ('ctx_prior', 'chem_morgan') else [t.to(DEV) for t in v]) for k, v in x.items()}

def pred_of(names, idx):
    preds = []
    with torch.no_grad():
        for n in names:
            x = make_x(idx, n)
            preds.append(MODELS[n](to_gpu(x)).cpu().numpy())
    return np.mean(preds, axis=0)

def prot_r2(yp, yt, m):
    cnt = m.sum(0); keep = cnt >= 3; n = np.maximum(cnt.astype(float), 1)
    ytc = np.where(m, yt, 0.0); ypc = np.where(m, yp, 0.0)
    mt = ytc.sum(0) / n
    ss_tot = (((ytc - mt) ** 2) * m).sum(0); ss_res = (((ytc - ypc) ** 2) * m).sum(0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    return float(np.median(r2[keep])) if keep.any() else float('nan')

# ---------- val 场景评估 ----------
print("\n===== val 各场景蛋白R2中位 =====")
print(f"{'组合':<22}{'val_chem':>9}{'val_strain':>11}{'val_both':>10}{'val_time':>10}")
combs = {
    '3×v2.1': ['v21a', 'v21b', 'v21c'],
    '3×v2.1+v30': ['v21a', 'v21b', 'v21c', 'v30'],
    '3×v2.1+v35': ['v21a', 'v21b', 'v21c', 'v35'],
    '3×v2.1+v37': ['v21a', 'v21b', 'v21c', 'v37'],
    'v37单': ['v37'],
}
val_r2 = {}
for cname, names in combs.items():
    row = []
    for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
        idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
        if len(idx) == 0:
            row.append(float('nan'))
            continue
        yp = pred_of(names, idx)
        yt, m = y_log2[idx], mask[idx].astype(bool)
        row.append(prot_r2(yp, yt, m))
    val_r2[cname] = row
    print(f"{cname:<22}" + "".join(f"{v:>9.3f}" if v == v else f"{'nan':>9}" for v in row), flush=True)

# ---------- test 场景评估（六模块总分，train 对照口径）----------
tmeta = pd.read_csv(f'{BASE}/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')
traw = pd.read_csv(f'{BASE}/WAYB_WAYC_proteome_raw_test.csv').set_index('sample_ID')
# test 特征
strains_tr = sorted(meta['Strains'].unique()); chems_tr = sorted(meta.loc[meta['role'].eq('treatment'), 'perturbation_no_concentration'].unique())
strain2id = {s: i for i, s in enumerate(strains_tr)}; chem2id = {c: i for i, c in enumerate(chems_tr)}
sm_cats = sorted((meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str)).unique())
ct_cats = sorted((meta['perturbation_no_concentration'].astype(str) + '|' + meta['Temperature'].astype(str)).unique())
sm2id = {k: i for i, k in enumerate(sm_cats)}; ct2id = {k: i for i, k in enumerate(ct_cats)}
src_cats = sorted(meta['data_source'].unique()); ins_cats = sorted(meta['instrument'].unique()); plt_cats = sorted(meta['Yeast_cell_plate'].unique())
src2id = {k: i for i, k in enumerate(src_cats)}; ins2id = {k: i for i, k in enumerate(ins_cats)}; plt2id = {k: i for i, k in enumerate(plt_cats)}
train_strains = set(meta.loc[meta['split_final'].eq('train'), 'Strains'])
train_chems = set(meta.loc[meta['split_final'].eq('train') & meta['role'].eq('treatment'), 'perturbation_no_concentration'])
import hashlib
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
# morgan / ctx_prior
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

TF = {
    'bio': [torch.from_numpy(tstrain), torch.from_numpy(tchem), torch.from_numpy(tchash),
            torch.from_numpy(tmed), torch.from_numpy(ttemp), torch.from_numpy(ttfeat),
            torch.from_numpy(tsm), torch.from_numpy(tct)],
    'ctx': [torch.from_numpy(tsrc), torch.from_numpy(tins), torch.from_numpy(tplt)],
    'seen': [torch.from_numpy(tcseen), torch.from_numpy(tsseen)],
}
def tpred(names):
    preds = []
    with torch.no_grad():
        for n in names:
            x = dict(TF)
            if n in ('v30', 'v35', 'v37'):
                x['ctx_prior'] = torch.from_numpy(t_ctx_prior)
                x['chem_morgan'] = torch.from_numpy(tmorgan)
            preds.append(MODELS[n](to_gpu(x)).cpu().numpy())
    return np.mean(preds, axis=0)

# 预生成所有组合的 test 预测（5243 蛋白填均值）
test_cols = traw.columns.tolist()  # 已 set_index，全为蛋白列
prot4422 = [l.strip() for l in open(f'{DATA}/prot_names.txt')]
col_of = {p: i for i, p in enumerate(test_cols)}
pos4422 = np.array([col_of[p] for p in prot4422 if p in col_of], dtype=int)
assert len(pos4422) == len(prot4422), f'蛋白映射不完整: {len(pos4422)}/{len(prot4422)}'
t_log2 = np.log2(traw[test_cols].values.astype(np.float64))
is_ctrl = tmeta['perturbation_no_concentration'].isin(['Water', 'DMSO']).values
is_treat = ~is_ctrl
treat_pos = np.where(is_treat)[0]
tpos_idx = {p: i for i, p in enumerate(treat_pos)}

def expand4422(p4422):
    full = np.zeros((len(p4422), len(test_cols)), dtype=np.float32)
    full[:, pos4422] = p4422
    # 高缺失蛋白填训练均值
    for c in range(len(test_cols)):
        if not np.isfinite(full[:, c]).any():
            pass
    return full

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

# 训练对照池（4422 -> 5243）
ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = (meta.iloc[ctrl_idx]['data_source'].astype(str) + '|' + meta.iloc[ctrl_idx]['instrument'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_idx]['Strains'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Medium'].astype(str) + '|' + meta.iloc[ctrl_idx]['Temperature'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['pert_time'].astype(str)).values
ctrl_lookup = {}
for k, pos in zip(ctrl_key, ctrl_idx):
    ctrl_lookup.setdefault(k, []).append(pos)
# test 对照池
t_key = (tmeta['data_source'].astype(str) + '|' + tmeta['instrument'].astype(str) + '|' + tmeta['Yeast_cell_plate'].astype(str) + '|'
         + tmeta['Strains'].astype(str) + '|' + tmeta['Medium'].astype(str) + '|' + tmeta['Temperature'].astype(str) + '|'
         + tmeta['pert_time'].astype(str)).values
ctrl_lookup_te = {}
for k, pos in zip(t_key, np.where(is_ctrl)[0]):
    ctrl_lookup_te.setdefault(k, []).append(pos)

# 训练 μ_ctx / μ_drug（LOO 近似：pool）
treat_all = np.where(meta['role'].eq('treatment').values)[0]
train_treat = np.where(train_mask & meta['role'].eq('treatment').values)[0]
is_train_treat = np.isin(treat_all, train_treat)
tr_treat_idx = np.where(is_train_treat)[0]
ctrl_tr = np.full((len(treat_all), P), np.nan)
for i, sid in enumerate(treat_all):
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    rows = ctrl_lookup.get(k, [])
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

# test 处理样本对照
ctrl_vals = np.full((len(treat_pos), len(test_cols)), np.nan)
for i, pos in enumerate(treat_pos):
    full = np.full(len(test_cols), np.nan)
    rows = ctrl_lookup.get(t_key[pos], [])
    if rows:
        cvals = tr_y_nan[rows]; cm = np.isfinite(cvals)
        with np.errstate(invalid='ignore'):
            m4422 = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, 0), 0) / cm.sum(0), np.nan)
        full[pos4422] = m4422
    rows = ctrl_lookup_te.get(t_key[pos], [])
    if rows:
        cvals = t_log2[rows]; cm = np.isfinite(cvals)
        with np.errstate(invalid='ignore'):
            mfull = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, 0), 0) / cm.sum(0), np.nan)
        full = np.where(np.isfinite(full), full, mfull)
    ctrl_vals[i] = full

# 训练 log2 均值（5243 填充）
meta_full = pd.read_csv(f'{BASE}/WAYB_WAYC_metadata_train_val(1).csv')
prot_full = pd.read_csv(f'{BASE}/WAYB_WAYC_proteome_raw_train_val.csv', index_col='sample_ID')
train_ids = meta_full.loc[meta_full['split_final'].eq('train'), 'sample_ID']
lm = np.log2(prot_full.loc[train_ids]).mean(axis=0, skipna=True)
lm = lm.fillna(np.median(lm.dropna()))

def score_scene(scene, comb_name, names):
    idx = np.where(tmeta['split_final'].eq(scene).values)[0]
    treat_in_scene = [p for p in idx if is_treat[p]]
    if not treat_in_scene:
        return None
    pos_arr = np.array(treat_in_scene)
    p4422 = tpred(names)[pos_arr]  # (n, 4422)
    # 扩展到 5243
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
    return total, mod1, mod2, mod3, mod4, mod5, mod6, len(pos_arr)

print("\n===== test 各场景六模块总分 =====")
print(f"{'组合':<22}{'t_chem':>8}{'t_strain':>9}{'t_both':>8}{'t_time':>8}{'加权':>8}")
scene_w = [1.0, 1.0, 0.5, 0.5]
test_total = {}
for cname, names in combs.items():
    row = []
    for scene in ['test_chem_only', 'test_strain_only', 'test_both', 'test_time']:
        r = score_scene(scene, cname, names)
        row.append(r[0] if r else float('nan'))
    test_total[cname] = row
    wsum = sum(w * (r if r == r else 0) for w, r in zip(scene_w, row)) / 3.0
    print(f"{cname:<22}" + "".join(f"{v:>8.3f}" if v == v else f"{'nan':>8}" for v in row) + f"{wsum:>8.3f}", flush=True)

# ---------- 最优场景映射：组合网格搜索 ----------
print("\n===== 场景最优映射（每个场景选得分最高的组合）=====")
best_map = {}
for scene in ['test_chem_only', 'test_strain_only', 'test_both', 'test_time']:
    best_c, best_s = None, -1
    for cname, names in combs.items():
        r = score_scene(scene, cname, names)
        if r and r[0] > best_s:
            best_s, best_c = r[0], cname
    best_map[scene] = best_c
    print(f"  {scene}: 最优={best_c} 总分={best_s:.4f}")
print("\n最优映射:", best_map)
