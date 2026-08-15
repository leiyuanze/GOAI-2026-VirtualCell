# -*- coding: utf-8 -*-
"""评估 3×v37 多 seed 集成 vs 单模型"""
import numpy as np, pandas as pd, torch, importlib.util, pickle

DATA = 'data'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float32)
mask = np.load(f'{DATA}/mask.npy').astype(np.float32)
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))
P = y_log2.shape[1]
ctx_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

_s = importlib.util.spec_from_file_location('m37', '04_model_v37.py')
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)

def load(p):
    m = _m.VCellModel(feats, P=P)
    m.load_state_dict(torch.load(f'{DATA}/{p}', map_location=DEV, weights_only=True))
    m.set_strain_avg()
    return m.to(DEV).eval()

m1 = load('model_v37_best.pt')
m2 = load('model_v37_s43_best.pt')
m3 = load('model_v37_s44_best.pt')
ENS = [m1, m2, m3]

def make_x(idx):
    return {
        'bio': [torch.from_numpy(feats['strain_id'][idx]), torch.from_numpy(feats['chem_id'][idx]),
                torch.from_numpy(feats['chem_hash'][idx]), torch.from_numpy(feats['medium_onehot'][idx]),
                torch.from_numpy(feats['temp_norm'][idx]), torch.from_numpy(feats['time_feat'][idx]),
                torch.from_numpy(feats['sm_id'][idx]), torch.from_numpy(feats['ct_id'][idx])],
        'ctx': [torch.from_numpy(feats['src_id'][idx]), torch.from_numpy(feats['ins_id'][idx]),
                torch.from_numpy(feats['plt_id'][idx])],
        'seen': [torch.from_numpy(feats['chem_seen'][idx]), torch.from_numpy(feats['strain_seen'][idx])],
        'ctx_prior': torch.from_numpy(ctx_all[idx]),
        'chem_morgan': torch.from_numpy(feats['chem_morgan'][idx]),
    }

def pred_of(models, idx):
    x = make_x(idx)
    xg = {k: (v.to(DEV) if k in ('ctx_prior', 'chem_morgan') else [t.to(DEV) for t in v]) for k, v in x.items()}
    ps = []
    with torch.no_grad():
        for m in models:
            ps.append(m(xg).cpu().numpy())
    return np.mean(ps, axis=0)

def prot_r2(yp, yt, m):
    cnt = m.sum(0); keep = cnt >= 3; n = np.maximum(cnt.astype(float), 1)
    ytc = np.where(m, yt, 0.0); ypc = np.where(m, yp, 0.0)
    mt = ytc.sum(0) / n
    ss_tot = (((ytc - mt) ** 2) * m).sum(0); ss_res = (((ytc - ypc) ** 2) * m).sum(0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    return float(np.median(r2[keep])) if keep.any() else float('nan')

print(f"{'场景':<18}{'v37单':>10}{'3xv37':>9}")
for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
    idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    yt, m = y_log2[idx], mask[idx].astype(bool)
    r1 = prot_r2(pred_of([m1], idx), yt, m)
    r3 = prot_r2(pred_of(ENS, idx), yt, m)
    print(f"{scene:<18}{r1:>10.3f}{r3:>9.3f}")
