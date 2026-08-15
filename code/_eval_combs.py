# -*- coding: utf-8 -*-
"""深入对比 val_both/val_strain 各组合（s44 作 v37 参考）"""
import numpy as np, pandas as pd, torch, importlib.util, pickle

DATA = 'data'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float32)
mask = np.load(f'{DATA}/mask.npy').astype(np.float32)
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))
P = y_log2.shape[1]
ctx_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def load(p, cls, sa=False):
    m = cls(feats, P=P)
    m.load_state_dict(torch.load(f'{DATA}/{p}', map_location=DEV, weights_only=True))
    if sa:
        m.set_strain_avg()
    return m.to(DEV).eval()

_s21 = importlib.util.spec_from_file_location('m21', '04_model_v21.py')
_m21 = importlib.util.module_from_spec(_s21); _s21.loader.exec_module(_m21)
_s30 = importlib.util.spec_from_file_location('m30', '04_model_v30.py')
_m30 = importlib.util.module_from_spec(_s30); _s30.loader.exec_module(_m30)
_s37 = importlib.util.spec_from_file_location('m37', '04_model_v37.py')
_m37 = importlib.util.module_from_spec(_s37); _s37.loader.exec_module(_m37)

MODELS = {
    'v21a': load('model_v21.pt', _m21.VCellModel),
    'v21b': load('model_v21_s43.pt', _m21.VCellModel),
    'v21c': load('model_v21_s44.pt', _m21.VCellModel),
    'v30': load('model_v30_best.pt', _m30.VCellModel, True),
    'v35': load('model_v35_best.pt', _m30.VCellModel, True),
    'v37': load('model_v37_s44_best.pt', _m37.VCellModel, True),
}

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

def t_gpu(x):
    return {k: (v.to(DEV) if k in ('ctx_prior', 'chem_morgan') else [t.to(DEV) for t in v]) for k, v in x.items()}

def pred_of(names, idx):
    ps = []
    with torch.no_grad():
        for n in names:
            ps.append(MODELS[n](t_gpu(make_x(idx, n))).cpu().numpy())
    return np.mean(ps, axis=0)

def prot_r2(yp, yt, m):
    cnt = m.sum(0); keep = cnt >= 3; n = np.maximum(cnt.astype(float), 1)
    ytc = np.where(m, yt, 0.0); ypc = np.where(m, yp, 0.0)
    mt = ytc.sum(0) / n
    ss_tot = (((ytc - mt) ** 2) * m).sum(0); ss_res = (((ytc - ypc) ** 2) * m).sum(0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    return float(np.median(r2[keep])) if keep.any() else float('nan')

combs = {
    'v37单(s44)': ['v37'],
    '3x21': ['v21a', 'v21b', 'v21c'],
    '3x21+v35': ['v21a', 'v21b', 'v21c', 'v35'],
    '3x21+v30': ['v21a', 'v21b', 'v21c', 'v30'],
    '3x21+v37': ['v21a', 'v21b', 'v21c', 'v37'],
    '3x21+v30+v35': ['v21a', 'v21b', 'v21c', 'v30', 'v35'],
    '3x21+2xv35': ['v21a', 'v21b', 'v21c', 'v35', 'v35'],
}
print(f"{'组合':<12}{'val_chem':>10}{'val_strain':>12}{'val_both':>10}{'val_time':>10}")
for cn, names in combs.items():
    row = []
    for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
        idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
        if len(idx) == 0:
            row.append(float('nan'))
            continue
        row.append(prot_r2(pred_of(names, idx), y_log2[idx], mask[idx].astype(bool)))
    print(f"{cn:<12}" + "".join(f"{v:>10.3f}" if v == v else f"{'nan':>10}" for v in row))
