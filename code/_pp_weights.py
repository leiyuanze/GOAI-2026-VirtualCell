# -*- coding: utf-8 -*-
"""每蛋白收缩融合：val 上算 v37/v5.2 per-protein PCC 权重，保存 npy（κ 网格）"""
import pickle
import numpy as np
import pandas as pd
import torch
import importlib.util

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy").astype(np.float32)
mask = np.load(f"{DATA}/mask.npy").astype(np.float32)
with open(f"{DATA}/feats.pkl", 'rb') as f:
    feats = pickle.load(f)
P = y_log2.shape[1]

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
    'chem_desc': torch.from_numpy(feats['chem_desc'].astype(np.float32)).to(DEV),
    'strain_dist_vec': torch.from_numpy(feats['strain_dist_vec'].astype(np.float32)).to(DEV),
    'chem_fp32': torch.from_numpy(feats['chem_fp32'].astype(np.float32)).to(DEV),
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
        'chem_desc': f['chem_desc'][idx],
        'strain_dist_vec': f['strain_dist_vec'][idx],
        'chem_fp32': f['chem_fp32'][idx],
    }

def load_model(mod_file, cls, ckpt, **kw):
    spec = importlib.util.spec_from_file_location("m", mod_file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    model = getattr(m, cls)(feats, P=P, **kw).to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    model.set_strain_avg(); model.eval()
    return model

m37 = load_model(r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v37.py",
                 'VCellModel', f"{DATA}/model_v37_42_best.pt")
U_basis = np.load(f"{DATA}/v50_response_basis.npy")
m52 = load_model(r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v50_v52.py",
                 'VCellModel', f"{DATA}/model_v50_42_best_v52.pt", response_basis=U_basis)

SCENES = ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']
val_idx = np.where(meta['split_final'].isin(SCENES).values & meta['role'].eq('treatment').values)[0]
with torch.no_grad():
    x = make_x(val_idx)
    p37 = m37(x).cpu().numpy()
    p52 = m52(x).cpu().numpy()
yt = y_log2[val_idx]
m = mask[val_idx].astype(bool)

def per_protein_pcc(pred, yt, m):
    pccs = np.full(P, np.nan)
    for pp in range(P):
        ok = m[:, pp] & np.isfinite(pred[:, pp])
        if ok.sum() >= 5:
            a, b = pred[:, pp][ok], yt[:, pp][ok]
            if a.std() > 1e-12 and b.std() > 1e-12:
                pccs[pp] = np.corrcoef(a, b)[0, 1]
    return pccs

pcc37 = per_protein_pcc(p37, yt, m)
pcc52 = per_protein_pcc(p52, yt, m)
n_obs = m.sum(0).astype(np.float32)
w_j = pcc37 / (pcc37 + pcc52 + 1e-6)
print(f"val 合并：v37 PCC 中位 {np.nanmedian(pcc37):.3f} / v5.2 {np.nanmedian(pcc52):.3f}")
print(f"w_j 均值 {np.nanmean(w_j):.3f}（>0.75 表示 v37 优势蛋白更多）")

for kappa in [100, 300, 500]:
    w_shrunk = (n_obs * w_j + kappa * 0.75) / (n_obs + kappa)
    w_shrunk = np.where(np.isfinite(w_shrunk), w_shrunk, 0.75)
    np.save(f"{DATA}/pp_weight_kappa{kappa}.npy", w_shrunk.astype(np.float32))
    print(f"kappa={kappa}: w_shrunk 均值 {w_shrunk.mean():.3f} / <0.75 蛋白占比 {(w_shrunk<0.74).mean()*100:.1f}%")

np.save(f"{DATA}/pp_nobs.npy", n_obs)
print("已保存 pp_weight_kappa{100,300,500}.npy + pp_nobs.npy")
