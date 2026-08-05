# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 08 v2.6 对比评估
v2.6 vs v2.1(s42) vs v2.5 vs 4-集成 → 四场景蛋白R²+FC PCC 对照表
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
ctx_prior_all = feats['ctx_prior'].astype(np.float32)

_s21 = importlib.util.spec_from_file_location("m21", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v21.py")
_m21 = importlib.util.module_from_spec(_s21); _s21.loader.exec_module(_m21)
_s25 = importlib.util.spec_from_file_location("m25", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v25.py")
_m25 = importlib.util.module_from_spec(_s25); _s25.loader.exec_module(_m25)
_s26 = importlib.util.spec_from_file_location("m26", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v26.py")
_m26 = importlib.util.module_from_spec(_s26); _s26.loader.exec_module(_m26)

def load_model(path, cls, needs_ctx=False):
    m = cls(feats, P=P)
    m.load_state_dict(torch.load(f"{DATA}/{path}", map_location=DEV, weights_only=True))
    return m.to(DEV).eval()

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
        d['ctx_prior'] = torch.from_numpy(ctx_prior_all[idx])
    return d

def eval_model(name, m, scene_idx, ctx=False):
    print(f"  {name}", flush=True)
    results = {}
    for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
        idx = scene_idx[scene]
        if len(idx) == 0:
            continue
        x = make_x(idx, with_ctx=ctx)
        with torch.no_grad():
            if ctx:
                pred = m({k: (v.to(DEV) if k == 'ctx_prior' else [t.to(DEV) for t in v]) for k, v in x.items()}).cpu().numpy()
            else:
                pred = m({k: [t.to(DEV) for t in v] for k, v in x.items()}).cpu().numpy()
        yt, mb = y_log2[idx], mask[idx].astype(bool)
        valid = mb & np.isfinite(pred)
        cnt = valid.sum(0); keep = cnt >= 3
        n = np.maximum(cnt.astype(float), 1)
        ytc = np.where(valid, yt, 0.0); ypc = np.where(valid, pred, 0.0)
        mt = ytc.sum(0) / n
        ss_tot = (((ytc - mt) ** 2) * valid).sum(0)
        ss_res = (((ytc - ypc) ** 2) * valid).sum(0)
        p2 = float(np.median(1 - ss_res / np.maximum(ss_tot, 1e-12)))
        # FC PCC
        yc = np.array([_matched_control_mean(s) for s in idx])
        fc_ok = np.isfinite(yc) & mb & np.isfinite(pred)
        d_pred = (pred - yc)[fc_ok]; d_true = (yt - yc)[fc_ok]
        fc = float(np.corrcoef(d_pred, d_true)[0, 1]) if len(d_pred) > 10 else float('nan')
        results[scene] = (p2, fc)
    return results

# matched control 复用
ctrl_idx_pos = np.where(meta['role'].eq('control').values)[0]
ctrl_key = (meta.iloc[ctrl_idx_pos]['data_source'].astype(str) + '|' + meta.iloc[ctrl_idx_pos]['instrument'].astype(str) + '|'
            + meta.iloc[ctrl_idx_pos]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_idx_pos]['Strains'].astype(str) + '|'
            + meta.iloc[ctrl_idx_pos]['Medium'].astype(str) + '|' + meta.iloc[ctrl_idx_pos]['Temperature'].astype(str) + '|'
            + meta.iloc[ctrl_idx_pos]['pert_time'].astype(str)).values
ctrl_lookup = {}
for k, pos in zip(ctrl_key, ctrl_idx_pos):
    ctrl_lookup.setdefault(k, []).append(pos)
tr_y_nan = np.where(mask.astype(bool), y_log2, np.nan)

def _matched_control_mean(sid):
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    if k not in ctrl_lookup:
        return np.full(P, np.nan)
    rows = ctrl_lookup[k]
    cvals = tr_y_nan[rows]; cm = mask[rows] > 0
    with np.errstate(invalid='ignore'):
        return np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)

scene_idx = {}
for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
    scene_idx[scene] = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]

print("=" * 70)
print("v2.6 对比评估")
print("=" * 70)

# 加载模型
m21 = load_model('model_v21.pt', _m21.VCellModel)
m25 = load_model('model_v25_best.pt', _m25.VCellModel)
m26 = load_model('model_v26_best.pt', _m26.VCellModel, needs_ctx=True)

res21 = eval_model("v2.1 (s42)", m21, scene_idx)
res25 = eval_model("v2.5", m25, scene_idx)
res26 = eval_model("v2.6 ★", m26, scene_idx, ctx=True)

# 汇总
print(f"\n{'场景':<20}{'v2.1蛋白R2':>11}{'v2.5蛋白R2':>11}{'v2.6蛋白R2':>11}{'v2.1 FC':>9}{'v2.5 FC':>9}{'v2.6 FC':>9}")
print("-" * 80)
for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
    p21, f21 = res21.get(scene, (0, 0))
    p25, f25 = res25.get(scene, (0, 0))
    p26, f26 = res26.get(scene, (0, 0))
    b21 = "↑" if p26 > p21 else "↓" if p26 < p21 else "="
    b25 = "↑" if p26 > p25 else "↓" if p26 < p25 else "="
    print(f"{scene:<20}{p21:>11.3f}{p25:>11.3f}{p26:>11.3f}{b21}{b25}  {f21:>9.3f}{f25:>9.3f}{f26:>9.3f}")

print("\n08 DONE", flush=True)
