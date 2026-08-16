# -*- coding: utf-8 -*-
"""
步骤16：多 seed 伪测试评估（gpt2 P0-3 / gpt1 评估补强）
对 v5.1 的 3 个 seed（42/43/44）评估：
  1) v5.1 单模型四场景
  2) 0.8*v37 + 0.2*v5.1 集成四场景
报告 mean±std（不报告单次最好值）
用法：python _pseudo_test.py
"""
import os, pickle, numpy as np, pandas as pd, torch
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
    'chem_desc': torch.from_numpy(feats['chem_desc'].astype(np.float32)).to(DEV),
    'strain_dist_vec': torch.from_numpy(feats['strain_dist_vec'].astype(np.float32)).to(DEV),
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
    }

_spec37 = importlib.util.spec_from_file_location("m37", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v37.py")
_m37 = importlib.util.module_from_spec(_spec37); _spec37.loader.exec_module(_m37)
_spec50 = importlib.util.spec_from_file_location("m50", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v50.py")
_m50 = importlib.util.module_from_spec(_spec50); _spec50.loader.exec_module(_m50)

U_basis = np.load(f"{DATA}/v50_response_basis.npy")
m37 = _m37.VCellModel(feats, P=P).to(DEV)
m37.load_state_dict(torch.load(f"{DATA}/model_v37_42_best.pt", map_location=DEV, weights_only=True))
m37.set_strain_avg(); m37.eval()

SCENES = ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']
def eval_pred(pred_dict):
    out = {}
    for scene in SCENES:
        idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
        if len(idx) == 0:
            out[scene] = (np.nan, np.nan)
            continue
        pred = pred_dict[scene]
        yt, m = y_log2[idx], mask[idx].astype(bool)
        valid = m & np.isfinite(pred)
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
        out[scene] = (p2, fc)
    return out

seeds = [1, 2, 42, 43, 44]  # gpt2 步骤16: for seed in [1,2,3,4,5]
results_v51 = {s: {} for s in seeds}
results_ens = {s: {} for s in seeds}
for sd in seeds:
    ck = f"{DATA}/model_v50_{sd}_best.pt"
    if not os.path.exists(ck):
        print(f"skip seed {sd}: {ck} 不存在", flush=True)
        continue
    m50 = _m50.VCellModel(feats, P=P, response_basis=U_basis).to(DEV)
    m50.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
    m50.set_strain_avg(); m50.eval()
    pd_all = {}
    for scene in SCENES:
        idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
        x = make_x(idx)
        with torch.no_grad():
            p50 = m50(x).cpu().numpy()
            p37 = m37(x).cpu().numpy()
        pd_all[scene] = (p50, p37)
    results_v51[sd] = eval_pred({s: pd_all[s][0] for s in SCENES})
    results_ens[sd] = eval_pred({s: 0.8 * pd_all[s][1] + 0.2 * pd_all[s][0] for s in SCENES})
    print(f"seed {sd} done", flush=True)

print("\n=== 步骤16 多 seed 伪测试（5 seeds）===")
print(f"{'场景':<18}{'v5.1 单 R2':>16}{'v5.1 FC':>14}{'集成 R2':>16}{'集成 FC':>14}")
for scene in SCENES:
    r51 = [results_v51[s][scene][0] for s in seeds if results_v51[s]]
    f51 = [results_v51[s][scene][1] for s in seeds if results_v51[s]]
    re_ = [results_ens[s][scene][0] for s in seeds if results_ens[s]]
    fe = [results_ens[s][scene][1] for s in seeds if results_ens[s]]
    if not r51:
        continue
    print(f"{scene:<18}{np.mean(r51):>9.3f}+-{np.std(r51):.3f}{np.mean(f51):>11.3f}+-{np.std(f51):.3f}"
          f"{np.mean(re_):>11.3f}+-{np.std(re_):.3f}{np.mean(fe):>10.3f}+-{np.std(fe):.3f}")
