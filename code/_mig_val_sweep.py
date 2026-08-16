# -*- coding: utf-8 -*-
"""
val 伪新化合物 α 敏感性验证（gpt2 步骤16 伪测试思想，合规：不用 test 真值）
对 val_chem_only 的 6 个伪新化合物（train 无 Δ），用 train 迁移池做融合，
扫描 α 看 FC PCC / 蛋白R² 变化 → 验证"高相似度→α 大"假设
用法：python _mig_val_sweep.py [checkpoint_v37]
"""
import sys, pickle, numpy as np, pandas as pd, torch
import importlib.util

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/model_v37_42_best.pt"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy").astype(np.float32)
mask = np.load(f"{DATA}/mask.npy").astype(np.float32)
feats = pickle.load(open(f"{DATA}/feats.pkl", 'rb'))
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values
tr_y_nan = np.where(mask.astype(bool), y_log2, np.nan)
pert2morgan = feats['pert2morgan64']

# ---- 迁移池（train-only）----
treat_all = np.where(meta['role'].eq('treatment').values)[0]
ctrl_idx = np.where(meta['role'].eq('control').values & train_mask)[0]
ctrl_key = (meta.iloc[ctrl_idx]['data_source'].astype(str) + '|' + meta.iloc[ctrl_idx]['instrument'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_idx]['Strains'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Medium'].astype(str) + '|' + meta.iloc[ctrl_idx]['Temperature'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['pert_time'].astype(str)).values
lookup = {}
for k, pos in zip(ctrl_key, ctrl_idx):
    lookup.setdefault(k, []).append(pos)
# ★ 评估对照用全量观测（官方 M1 口径），迁移池仍 train-only
ctrl_idx_all = np.where(meta['role'].eq('control').values)[0]
ctrl_key_all = (meta.iloc[ctrl_idx_all]['data_source'].astype(str) + '|' + meta.iloc[ctrl_idx_all]['instrument'].astype(str) + '|'
                + meta.iloc[ctrl_idx_all]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_idx_all]['Strains'].astype(str) + '|'
                + meta.iloc[ctrl_idx_all]['Medium'].astype(str) + '|' + meta.iloc[ctrl_idx_all]['Temperature'].astype(str) + '|'
                + meta.iloc[ctrl_idx_all]['pert_time'].astype(str)).values
lookup_all = {}
for k, pos in zip(ctrl_key_all, ctrl_idx_all):
    lookup_all.setdefault(k, []).append(pos)
ctrl_all = np.full((len(treat_all), P), np.nan)
ctrl_all_eval = np.full((len(treat_all), P), np.nan)
for i, sid in enumerate(treat_all):
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    for lu, out in [(lookup, ctrl_all), (lookup_all, ctrl_all_eval)]:
        rows = lu.get(k, [])
        if rows:
            cvals = tr_y_nan[rows]; cm = mask[rows] > 0
            with np.errstate(invalid='ignore'):
                out[i] = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)
delta_tr = tr_y_nan[treat_all] - ctrl_all
chem_of = meta['perturbation_no_concentration'].values[treat_all]
is_train = np.isin(treat_all, np.where(train_mask & meta['role'].eq('treatment').values)[0])
tr_strain = meta['Strains'].values[treat_all]
drug_strain_mu = {}
for i in np.where(is_train)[0]:
    key = (tr_strain[i], chem_of[i])
    drug_strain_mu.setdefault(key, []).append(delta_tr[i])
for k in drug_strain_mu:
    drug_strain_mu[k] = np.nanmean(np.stack(drug_strain_mu[k]), axis=0)
train_chems = sorted(set(chem_of[is_train]))

def topk_sim(test_c, k=5):
    v = pert2morgan.get(test_c)
    if v is None:
        return []
    sims = []
    for tc in train_chems:
        w = pert2morgan.get(tc)
        if w is None:
            continue
        s = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-8)
        sims.append((float(s), tc))
    sims.sort(reverse=True)
    return sims[:k]

# ---- 模型（v37）----
_spec = importlib.util.spec_from_file_location("m37", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v37.py")
_m37 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m37)
model = _m37.VCellModel(feats, P=P).to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True))
model.set_strain_avg(); model.eval()

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

# ---- val 伪新化合物样本 ----
pos_of = {sid: i for i, sid in enumerate(treat_all)}
idx = np.where(meta['split_final'].eq('val_chem_only').values & meta['role'].eq('treatment').values)[0]
with torch.no_grad():
    pred = model(make_x(idx)).cpu().numpy()
yc = ctrl_all_eval[[pos_of[s] for s in idx]]
is_new = np.array([meta['perturbation_no_concentration'].values[s] not in set(train_chems) for s in idx])
new_idx = idx[is_new]
print(f"val_chem_only 处理样本 {len(idx)}，伪新化合物样本 {len(new_idx)}")

# 逐化合物统计
new_chems = sorted(set(meta['perturbation_no_concentration'].values[new_idx]))
for c in new_chems:
    sel = np.where(meta['perturbation_no_concentration'].values[new_idx] == c)[0]
    print(f"\n=== {c} (n={len(sel)}, top1_sim={topk_sim(c)[0][0]:.3f}) ===")
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        tops = topk_sim(c)
        mus, ws = [], []
        for sim, tc in tops:
            mu = drug_strain_mu.get((meta['Strains'].values[new_idx[sel[0]]], tc))
            if mu is None:
                continue
            mus.append(mu); ws.append(max(sim, 0.0))
        if not mus:
            print(f"  (无迁移候选: {tops})")
            break
        ws = np.array(ws); ws = ws / ws.sum()
        mig_delta = np.sum(np.stack([w * m for w, m in zip(ws, mus)]), axis=0)
        # ★ NaN 处理：迁移 Δ 只在有限值蛋白上生效
        mig_delta = np.where(np.isfinite(mig_delta), mig_delta, 0.0)
        # 逐样本融合
        p = pred[is_new][sel].copy(); yc_s = yc[is_new][sel]
        ok = np.isfinite(yc_s) & np.isfinite(p) & np.isfinite(y_log2[new_idx[sel]])
        model_delta = p - yc_s
        fused = yc_s + (1 - alpha) * model_delta + alpha * mig_delta[None, :]
        # FC PCC（样本维度全局）
        d_pred = (fused - yc_s)[ok]; d_true = (y_log2[new_idx[sel]] - yc_s)[ok]
        if len(d_pred) > 10:
            fc = np.corrcoef(d_pred, d_true)[0, 1]
        else:
            fc = float('nan')
        # 蛋白 R²（相对 yc，ok 内）
        n = ok.sum()
        if n > 0:
            r2 = 1 - ((y_log2[new_idx[sel]][ok] - fused[ok]) ** 2).sum() / max(((y_log2[new_idx[sel]][ok] - yc_s[ok]) ** 2).sum(), 1e-12)
        else:
            r2 = float('nan')
        print(f"  alpha={alpha:.1f}  FC PCC={fc:.4f}  ΔR²(vs yc)={r2:.4f}  (n={n})")
