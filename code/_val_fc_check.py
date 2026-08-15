# -*- coding: utf-8 -*-
"""验证：用 val 集跑 _test_score 的 M1 逻辑，与已知 val FC 对比，确认口径正确"""
import numpy as np, pandas as pd, torch, importlib.util, pickle

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float64)
mask = np.load(f'{DATA}/mask.npy').astype(bool)
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values
tr_y_nan = np.where(mask, y_log2, np.nan)
ctx_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)

# 加载 v3.7
_s = importlib.util.spec_from_file_location('m37', r'D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v37.py')
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
model = _m.VCellModel(feats, P=P)
model.load_state_dict(torch.load(f'{DATA}/model_v37_best.pt', map_location=DEV, weights_only=True))
model.to(DEV).eval()
model.set_strain_avg()

def make_x(idx):
    return {
        'bio': [torch.from_numpy(feats['strain_id'][idx]), torch.from_numpy(feats['chem_id'][idx]),
                torch.from_numpy(feats['chem_hash'][idx]), torch.from_numpy(feats['medium_onehot'][idx]),
                torch.from_numpy(feats['temp_norm'][idx]), torch.from_numpy(feats['time_feat'][idx]),
                torch.from_numpy(feats['sm_id'][idx]), torch.from_numpy(feats['ct_id'][idx])],
        'ctx': [torch.from_numpy(feats['src_id'][idx]), torch.from_numpy(feats['ins_id'][idx]), torch.from_numpy(feats['plt_id'][idx])],
        'seen': [torch.from_numpy(feats['chem_seen'][idx]), torch.from_numpy(feats['strain_seen'][idx])],
        'ctx_prior': torch.from_numpy(ctx_all[idx]),
        'chem_morgan': torch.from_numpy(feats['chem_morgan'][idx]),
    }

def mk_key(df):
    return (df['data_source'].astype(str) + '|' + df['instrument'].astype(str) + '|'
            + df['Yeast_cell_plate'].astype(str) + '|' + df['Strains'].astype(str) + '|'
            + df['Medium'].astype(str) + '|' + df['Temperature'].astype(str) + '|'
            + df['pert_time'].astype(str)).values

# 训练对照池（4422 蛋白）
ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = mk_key(meta.iloc[ctrl_idx])  # 只用对照行
ctrl_lookup = {}
for k, pos in zip(ctrl_key, ctrl_idx):
    ctrl_lookup.setdefault(k, []).append(pos)

def safe_pcc(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10: return 0.0
    a, b = a[ok], b[ok]
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0

print(f"{'场景':<16}{'样本':>6}{'FC_PCC(对照池)':>14}{'FC_PCC(值数)':>12}")
for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
    idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    if len(idx) == 0:
        continue
    x = make_x(idx)
    xg = {k: (v.to(DEV) if k in ('ctx_prior', 'chem_morgan') else [t.to(DEV) for t in v]) for k, v in x.items()}
    with torch.no_grad():
        pred = model(xg).cpu().numpy()
    yt = y_log2[idx]
    m = mask[idx].astype(bool)
    # matched control
    yc = np.full((len(idx), P), np.nan)
    for i, sid in enumerate(idx):
        r = meta.iloc[sid]
        k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
             + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
        rows = ctrl_lookup.get(k, [])
        if not rows:
            continue
        cvals = tr_y_nan[rows]
        cm = np.isfinite(cvals)
        with np.errstate(invalid='ignore'):
            yc[i] = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, 0), 0) / cm.sum(0), np.nan)
    dp = pred - yc
    dt = yt - yc
    ok = np.isfinite(dp) & np.isfinite(dt) & m
    fc = safe_pcc(dp[ok].ravel(), dt[ok].ravel())
    print(f"{scene:<16}{len(idx):>6}{fc:>14.3f}{ok.sum():>12}")
