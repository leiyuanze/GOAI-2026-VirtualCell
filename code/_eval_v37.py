# -*- coding: utf-8 -*-
"""
v37 独立评估（干净 feats + 全量观测对照 FC）
用法：python _eval_v37.py [checkpoint_path]
"""
import sys, pickle, numpy as np, pandas as pd
import torch
import importlib.util

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/model_v37_42_best.pt"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy").astype(np.float32)
mask = np.load(f"{DATA}/mask.npy").astype(np.float32)
with open(f"{DATA}/feats.pkl", 'rb') as f:
    feats = pickle.load(f)
P = y_log2.shape[1]
tr_y_nan = np.where(mask.astype(bool), y_log2, np.nan)

treat_all = np.where(meta['role'].eq('treatment').values)[0]
ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = (meta.iloc[ctrl_idx]['data_source'].astype(str) + '|' + meta.iloc[ctrl_idx]['instrument'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_idx]['Strains'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Medium'].astype(str) + '|' + meta.iloc[ctrl_idx]['Temperature'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['pert_time'].astype(str)).values
lookup = {}
for k, pos in zip(ctrl_key, ctrl_idx):
    lookup.setdefault(k, []).append(pos)
ctrl_all = np.full((len(treat_all), P), np.nan, dtype=np.float32)
for i, sid in enumerate(treat_all):
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    rows = lookup.get(k, [])
    if rows:
        cvals = tr_y_nan[rows]; cm = mask[rows] > 0
        with np.errstate(invalid='ignore'):
            ctrl_all[i] = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)
pos_of = {sid: i for i, sid in enumerate(treat_all)}

ctx_prior_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)
_feat_g = {
    'strain_id': torch.from_numpy(feats['strain_id']).to(DEV),
    'chem_id': torch.from_numpy(feats['chem_id']).to(DEV),
    'chem_hash': torch.from_numpy(feats['chem_hash']).to(DEV),
    'medium_onehot': torch.from_numpy(feats['medium_onehot']).to(DEV),
    'temp_norm': torch.from_numpy(feats['temp_norm']).to(DEV),
    'time_feat': torch.from_numpy(feats['time_feat']).to(DEV),
    'sm_id': torch.from_numpy(feats['sm_id']).to(DEV),
    'ct_id': torch.from_numpy(feats['ct_id']).to(DEV),
    'src_id': torch.from_numpy(feats['src_id']).to(DEV),
    'ins_id': torch.from_numpy(feats['ins_id']).to(DEV),
    'plt_id': torch.from_numpy(feats['plt_id']).to(DEV),
    'chem_seen': torch.from_numpy(feats['chem_seen']).to(DEV),
    'strain_seen': torch.from_numpy(feats['strain_seen']).to(DEV),
    'ctx_prior': torch.from_numpy(ctx_prior_all).to(DEV),
    'chem_morgan': torch.from_numpy(feats['chem_morgan']).to(DEV),
}

def make_x(idx):
    f = _feat_g
    return {
        'bio': [f['strain_id'][idx], f['chem_id'][idx], f['chem_hash'][idx],
                f['medium_onehot'][idx], f['temp_norm'][idx], f['time_feat'][idx],
                f['sm_id'][idx], f['ct_id'][idx]],
        'ctx': [f['src_id'][idx], f['ins_id'][idx], f['plt_id'][idx]],
        'seen': [f['chem_seen'][idx], f['strain_seen'][idx]],
        'ctx_prior': f['ctx_prior'][idx],
        'chem_morgan': f['chem_morgan'][idx],
    }

_spec = importlib.util.spec_from_file_location("m37", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v37.py")
_m37 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m37)
model = _m37.VCellModel(feats, P=P).to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV))
model.set_strain_avg()
model.eval()

print(f"=== v37 评估 {CKPT.split('/')[-1]} ===")
print(f"{'场景':<20}{'样本':>5}{'RMSE':>7}{'GlobalR2':>9}{'蛋白R2中位':>10}{'FC PCC':>8}")
for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
    idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    if len(idx) == 0:
        continue
    with torch.no_grad():
        pred = model(make_x(idx)).cpu().numpy()
    yt, m = y_log2[idx], mask[idx].astype(bool)
    valid = m & np.isfinite(pred)
    rmse = float(np.sqrt(((yt[m] - pred[m]) ** 2).mean()))
    a, b = yt[valid], pred[valid]
    g2 = 1 - ((a - b) ** 2).sum() / max(((a - a.mean()) ** 2).sum(), 1e-12)
    cnt = valid.sum(0); keep = cnt >= 3
    n = np.maximum(cnt.astype(float), 1)
    ytc = np.where(valid, yt, 0.0); ypc = np.where(valid, pred, 0.0)
    mt = ytc.sum(0) / n
    ss_tot = (((ytc - mt) ** 2) * valid).sum(0)
    ss_res = (((ytc - ypc) ** 2) * valid).sum(0)
    p2 = float(np.median(1 - ss_res / np.maximum(ss_tot, 1e-12)))
    yc = ctrl_all[[pos_of[s] for s in idx]]
    fc_ok = np.isfinite(yc) & m & np.isfinite(pred)
    d_pred = (pred - yc)[fc_ok]; d_true = (yt - yc)[fc_ok]
    fc = float(np.corrcoef(d_pred, d_true)[0, 1]) if len(d_pred) > 10 else float('nan')
    print(f"{scene:<20}{len(idx):>5}{rmse:>7.3f}{g2:>9.3f}{p2:>10.3f}{fc:>8.3f}")
