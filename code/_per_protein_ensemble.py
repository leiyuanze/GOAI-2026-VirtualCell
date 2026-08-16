# -*- coding: utf-8 -*-
"""
每蛋白收缩融合（gpt3 §四 / opus1 5.2）：
在 val 四场景上计算每个蛋白 v37 vs v5.2 的预测-真值 PCC，
w_j = PCC37/(PCC37+PCC52+eps)，收缩 w_j^shrunk = (n_j*w_j + kappa*w0)/(n_j+kappa)，
按蛋白融合生成 test 提交 → 六模块自评，对比全局 0.75/0.25。
"""
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
tr_y_nan = np.where(mask.astype(bool), y_log2, np.nan)
train_mask = meta['split_final'].eq('train').values

# ---------- 对照（全量观测，评估口径） ----------
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

# ---------- 模型加载 ----------
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
m52 = load_model(r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v50.py",
                 'VCellModel', f"{DATA}/model_v50_42_best.pt", response_basis=U_basis)

SCENES = ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']
with torch.no_grad():
    # val 四场景合并预测（用于算 per-protein PCC）
    val_idx = np.where(meta['split_final'].isin(SCENES).values & meta['role'].eq('treatment').values)[0]
    x = make_x(val_idx)
    p37_val = m37(x).cpu().numpy()
    p52_val = m52(x).cpu().numpy()
    yt_val = y_log2[val_idx]
    m_val = mask[val_idx].astype(bool)

# ---------- per-protein PCC（val，仅观测>3 的蛋白） ----------
def per_protein_pcc(pred, yt, m):
    pccs = np.full(P, np.nan)
    for pp in range(P):
        ok = m[:, pp] & np.isfinite(pred[:, pp])
        if ok.sum() >= 5:
            a, b = pred[:, pp][ok], yt[:, pp][ok]
            if a.std() > 1e-12 and b.std() > 1e-12:
                pccs[pp] = np.corrcoef(a, b)[0, 1]
    return pccs

pcc37 = per_protein_pcc(p37_val, yt_val, m_val)
pcc52 = per_protein_pcc(p52_val, yt_val, m_val)
n_obs = m_val.sum(0)
w_j = pcc37 / (pcc37 + pcc52 + 1e-6)
print(f"per-protein PCC: v37 中位 {np.nanmedian(pcc37):.3f} / v5.2 中位 {np.nanmedian(pcc52):.3f}")
print(f"w_j 原始：均值 {np.nanmean(w_j):.3f}（>0.75 表示 v37 蛋白更多）")

# ---------- test 预测 ----------
INPUT = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"
tmeta = pd.read_csv(f"{INPUT}/WAYB_WAYC_metadata_test(1).csv").set_index('sample_ID')
traw = pd.read_csv(f"{INPUT}/WAYB_WAYC_proteome_raw_test.csv").set_index('sample_ID')
sub = pd.read_csv(f"{DATA}/prediction_v50ens_base.csv", index_col=0)
cols = sub.columns.tolist()
prot4422 = [l.strip() for l in open(f"{DATA}/prot_names.txt")]
col_of = {p: i for i, p in enumerate(cols)}
pos4422 = np.array([col_of[p] for p in prot4422], dtype=int)

# test 特征（复用 07n 的加载——从 feats 取 test_* 字段）
# 由于 feats 里已含 test 特征（07n 用过），这里直接构造
from numpy import asarray
t_strain = np.full(len(tmeta), 0, dtype=np.int64)  # 占位（实际 07n 用 feats['test_*']）
# 直接读 feats 中 test 特征（07n 的逻辑：tstrain/tchem 等）
has_test_feats = all(k in feats for k in ['test_strain_id', 'test_chem_id'])
print("feats 含 test_* 特征:", has_test_feats)
if has_test_feats:
    def tmake_x(idx=None):
        N = len(tmeta)
        return {
            'bio': [torch.from_numpy(feats['test_strain_id']).to(DEV), torch.from_numpy(feats['test_chem_id']).to(DEV),
                    torch.from_numpy(feats['test_chem_hash']).to(DEV), torch.from_numpy(feats['test_medium_onehot']).to(DEV),
                    torch.from_numpy(feats['test_temp_norm']).to(DEV), torch.from_numpy(feats['test_time_feat']).to(DEV),
                    torch.from_numpy(feats['test_sm_id']).to(DEV), torch.from_numpy(feats['test_ct_id']).to(DEV)],
            'ctx': [torch.from_numpy(feats['test_src_id']).to(DEV), torch.from_numpy(feats['test_ins_id']).to(DEV),
                    torch.from_numpy(feats['test_plt_id']).to(DEV)],
            'seen': [torch.from_numpy(feats['test_chem_seen']).to(DEV), torch.from_numpy(feats['test_strain_seen']).to(DEV)],
            'ctx_prior': torch.from_numpy(np.nan_to_num(feats['test_ctx_prior'].astype(np.float32), nan=0.0)).to(DEV),
            'chem_morgan': torch.from_numpy(feats['test_chem_morgan']).to(DEV),
            'chem_desc': torch.from_numpy(feats['test_chem_desc'].astype(np.float32)).to(DEV),
            'strain_dist_vec': torch.from_numpy(feats['test_strain_dist_vec'].astype(np.float32)).to(DEV),
            'chem_fp32': torch.from_numpy(feats['test_chem_fp32'].astype(np.float32)).to(DEV),
        }
    with torch.no_grad():
        xt = tmake_x()
        p37_test = m37(xt).cpu().numpy()
        p52_test = m52(xt).cpu().numpy()
    # 填到提交矩阵（4422 位置）
    out = sub.values.astype(np.float64).copy()
    for kappa in [100, 300, 500]:
        w_shrunk = (n_obs * w_j + kappa * 0.75) / (n_obs + kappa)
        w_shrunk = np.where(np.isfinite(w_shrunk), w_shrunk, 0.75)
        fused = w_shrunk[None, :] * p37_test + (1 - w_shrunk)[None, :] * p52_test
        o = out.copy()
        o[:, pos4422] = fused
        np.savetxt(f"{DATA}/pred_pp_kappa{kappa}.csv", o, delimiter=',', comments='')
        print(f"kappa={kappa}: 已生成 pred_pp_kappa{kappa}.csv")
else:
    print("feats 无 test_* 特征，改用 07n 的加载路径（跳过，待 07n 改造）")
