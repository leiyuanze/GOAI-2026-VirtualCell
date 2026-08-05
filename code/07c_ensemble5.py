# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 07c 集成 v2.7：3×v2.1(s42/43/44) + v2.7(s42)
评估+提交，v2.7 替换 v2.5
"""
import numpy as np, pandas as pd, pickle, torch, importlib.util, hashlib

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
BASE = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy").astype(np.float32)
mask = np.load(f"{DATA}/mask.npy").astype(np.float32)
feats = pickle.load(open(f"{DATA}/feats.pkl", 'rb'))
P = y_log2.shape[1]
ctx_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)
with open(f"{DATA}/prot_names.txt", encoding='utf-8') as f:
    prot_names = f.read().splitlines()

_s21 = importlib.util.spec_from_file_location("m21", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v21.py")
_m21 = importlib.util.module_from_spec(_s21); _s21.loader.exec_module(_m21)
_s27 = importlib.util.spec_from_file_location("m27", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v27.py")
_m27 = importlib.util.module_from_spec(_s27); _s27.loader.exec_module(_m27)

def load_model(path, cls, set_avg=False):
    m = cls(feats, P=P)
    m.load_state_dict(torch.load(f"{DATA}/{path}", map_location=DEV, weights_only=True))
    if set_avg:
        m.set_strain_avg()
    return m.to(DEV).eval()

models = [
    (load_model('model_v21.pt', _m21.VCellModel), 'v21'),
    (load_model('model_v21_s43.pt', _m21.VCellModel), 'v21'),
    (load_model('model_v21_s44.pt', _m21.VCellModel), 'v21'),
    (load_model('model_v27_best.pt', _m27.VCellModel, set_avg=True), 'v27'),
]
print(f"[集成] {len(models)} 模型加载完成", flush=True)

def make_x(idx, with_ctx=False):
    d = {
        'bio': [torch.from_numpy(feats['strain_id'][idx]), torch.from_numpy(feats['chem_id'][idx]),
                torch.from_numpy(feats['chem_hash'][idx]), torch.from_numpy(feats['medium_onehot'][idx]),
                torch.from_numpy(feats['temp_norm'][idx]), torch.from_numpy(feats['time_feat'][idx]),
                torch.from_numpy(feats['sm_id'][idx]), torch.from_numpy(feats['ct_id'][idx])],
        'ctx': [torch.from_numpy(feats['src_id'][idx]), torch.from_numpy(feats['ins_id'][idx]),
                torch.from_numpy(feats['plt_id'][idx])],
        'seen': [torch.from_numpy(feats['chem_seen'][idx]), torch.from_numpy(feats['strain_seen'][idx])],
    }
    if with_ctx:
        d['ctx_prior'] = torch.from_numpy(ctx_all[idx])
    return d

def ensemble_pred(idx):
    preds = []
    with torch.no_grad():
        for m, tag in models:
            x = make_x(idx, with_ctx=(tag == 'v27'))
            if tag == 'v27':
                x_gpu = {k: (v.to(DEV) if k == 'ctx_prior' else [t.to(DEV) for t in v]) for k, v in x.items()}
            else:
                x_gpu = {k: [t.to(DEV) for t in v] for k, v in x.items()}
            preds.append(m(x_gpu).cpu().numpy())
    return np.mean(preds, axis=0)

# ---------- matched control ----------
ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = (meta.iloc[ctrl_idx]['data_source'].astype(str) + '|' + meta.iloc[ctrl_idx]['instrument'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_idx]['Strains'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Medium'].astype(str) + '|' + meta.iloc[ctrl_idx]['Temperature'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['pert_time'].astype(str)).values
ctrl_lookup = {}
for k, pos in zip(ctrl_key, ctrl_idx):
    ctrl_lookup.setdefault(k, []).append(pos)
tr_y_nan = np.where(mask.astype(bool), y_log2, np.nan)

def matched_control_mean(sid):
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    if k not in ctrl_lookup:
        return None
    rows = ctrl_lookup[k]
    cvals = tr_y_nan[rows]; cm = mask[rows] > 0
    with np.errstate(invalid='ignore'):
        return np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)

# ---------- val ----------
print(f"\n[新集成 3×v2.1+v2.7] {'场景':<16}{'样本':>5}{'RMSE':>7}{'GlobalR2':>9}{'蛋白R2中位':>10}{'FC PCC':>8}", flush=True)
for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
    idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    if len(idx) == 0:
        continue
    pred = ensemble_pred(idx)
    yt, m = y_log2[idx], mask[idx].astype(bool)
    valid = m & np.isfinite(pred)
    rmse = float(np.sqrt(((yt[m] - pred[m]) ** 2).mean()))
    a, b = yt[valid], pred[valid]
    g2 = 1 - ((a - b) ** 2).sum() / max(((a - a.mean()) ** 2).sum(), 1e-12)
    cnt = valid.sum(0); keep = cnt >= 3; n = np.maximum(cnt.astype(float), 1)
    ytc = np.where(valid, yt, 0.0); ypc = np.where(valid, pred, 0.0)
    mt = ytc.sum(0) / n
    ss_tot = (((ytc - mt) ** 2) * valid).sum(0); ss_res = (((ytc - ypc) ** 2) * valid).sum(0)
    p2 = float(np.median(1 - ss_res / np.maximum(ss_tot, 1e-12)))
    yc = np.array([matched_control_mean(s) for s in idx])
    fc_ok = np.isfinite(yc) & m & np.isfinite(pred)
    d_pred = (pred - yc)[fc_ok]; d_true = (yt - yc)[fc_ok]
    fc = float(np.corrcoef(d_pred, d_true)[0, 1]) if len(d_pred) > 10 else float('nan')
    print(f"{scene:<20}{len(idx):>5}{rmse:>7.3f}{g2:>9.3f}{p2:>10.3f}{fc:>8.3f}", flush=True)

# ---------- test ----------
tmeta = pd.read_csv(f"{BASE}/WAYB_WAYC_metadata_test(1).csv").set_index('sample_ID')
strains_tr = sorted(meta['Strains'].unique()); chems_tr = sorted(meta.loc[meta['role'].eq('treatment'), 'perturbation_no_concentration'].unique())
strain2id = {s: i for i, s in enumerate(strains_tr)}; chem2id = {c: i for i, c in enumerate(chems_tr)}
med2id = {m: i for i, m in enumerate(sorted(meta['Medium'].unique()))}
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

# test ctx_prior: 按训练集 (strain|medium|temp|time) 分组均值，unknown → gmean
gmean = feats['gmean']
ctx_key_tr = (meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|'
              + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)).values
tr_y_nan_full = np.where(mask.astype(bool), y_log2, np.nan)
train_mask_arr = meta['split_final'].eq('train').values
t_ctx_prior = np.tile(gmean, (len(tmeta), 1)).astype(np.float32)
ctx_grp = {}
for i in np.where(train_mask_arr)[0]:
    k = ctx_key_tr[i]
    ctx_grp.setdefault(k, []).append(tr_y_nan_full[i])
for k, vals in ctx_grp.items():
    ctx_grp[k] = np.nanmean(vals, axis=0)
t_ctx = (tmeta['Strains'].astype(str) + '|' + tmeta['Medium'].astype(str) + '|'
         + tmeta['Temperature'].astype(str) + '|' + tmeta['pert_time'].astype(str)).values
for i, k in enumerate(t_ctx):
    if k in ctx_grp:
        t_ctx_prior[i] = ctx_grp[k]
t_ctx_prior = np.nan_to_num(t_ctx_prior, nan=0.0)

# Ensemble predict
preds = []
for m, tag in models:
    x = {'bio': [torch.from_numpy(tstrain), torch.from_numpy(tchem), torch.from_numpy(tchash),
                 torch.from_numpy(tmed), torch.from_numpy(ttemp), torch.from_numpy(ttfeat),
                 torch.from_numpy(tsm), torch.from_numpy(tct)],
         'ctx': [torch.from_numpy(tsrc), torch.from_numpy(tins), torch.from_numpy(tplt)],
         'seen': [torch.from_numpy(tcseen), torch.from_numpy(tsseen)]}
    if tag == 'v27':
        x['ctx_prior'] = torch.from_numpy(t_ctx_prior)
        x_gpu = {k: (v.to(DEV) if k == 'ctx_prior' else [t.to(DEV) for t in v]) for k, v in x.items()}
    else:
        x_gpu = {k: [t.to(DEV) for t in v] for k, v in x.items()}
    with torch.no_grad():
        preds.append(m(x_gpu).cpu().numpy())
pred = np.mean(preds, axis=0)
sub = pd.DataFrame(pred, index=tmeta.index, columns=prot_names)
sub.index.name = 'sample_ID'
sub.to_csv(f"{DATA}/prediction_ensemble5.csv")
assert len(sub) == 4454 and sub.shape[1] == 4422
assert not sub.isna().any().any() and np.isfinite(sub.values).all()
print("\n[提交] prediction_ensemble5.csv 生成并校验通过", flush=True)
