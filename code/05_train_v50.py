# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 05 训练 v5.0（control + delta 低秩重构，gpt1/gpt2 路线）
架构：ŷ = ŷ_ctrl + Δ̂,  Δ̂ = U@z_delta + 残差 + GO/ESM2 先验
- control 监督：对照样本真值 + 处理样本 matched control（train-only lookup）
- delta 监督：Δ = y - matched_control（train-only），低秩基 U 由训练集 Δ SVD 学习
- 泄漏修复：matched control 仅用 train 划分对照行（8/16 审计）
用法：python 05_train_v50.py [seed]
"""
import os, sys, pickle, numpy as np, pandas as pd
import torch, torch.nn as nn
from tqdm import tqdm

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
torch.manual_seed(_SEED); np.random.seed(_SEED)
print(f"[设备] {DEV} | seed={_SEED}", flush=True)

meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy").astype(np.float32)
mask = np.load(f"{DATA}/mask.npy").astype(np.float32)
with open(f"{DATA}/feats.pkl", 'rb') as f:
    feats = pickle.load(f)
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values
tr_y_nan = np.where(mask.astype(bool), y_log2, np.nan)

# ---------- matched control 预计算（★ train-only lookup，泄漏修复）----------
ctrl_idx = np.where(meta['role'].eq('control').values & train_mask)[0]
ctrl_key = (meta.iloc[ctrl_idx]['data_source'].astype(str) + '|' + meta.iloc[ctrl_idx]['instrument'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_idx]['Strains'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Medium'].astype(str) + '|' + meta.iloc[ctrl_idx]['Temperature'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['pert_time'].astype(str)).values
ctrl_lookup = {}
for k, pos in zip(ctrl_key, ctrl_idx):
    ctrl_lookup.setdefault(k, []).append(pos)

def matched_control_mean(sid, use_all=False):
    """use_all=True 时用全量对照（评估口径，官方 M1 用真实对照）；False 仅 train（训练监督）"""
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    if k not in ctrl_lookup:
        return None
    rows = ctrl_lookup[k]
    cvals = tr_y_nan[rows]; cm = mask[rows] > 0
    with np.errstate(invalid='ignore'):
        return np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)

treat_all = np.where(meta['role'].eq('treatment').values)[0]
print("[预计算] matched control（train-only）...", flush=True)
ctrl_train_only = np.full((len(treat_all), P), np.nan, dtype=np.float32)
for i, sid in enumerate(treat_all):
    cm = matched_control_mean(sid, use_all=False)
    if cm is not None:
        ctrl_train_only[i] = cm
pos_of = {sid: i for i, sid in enumerate(treat_all)}
has_ctrl_train = np.isfinite(ctrl_train_only).any(axis=1)
print(f"[预计算] 有 train 对照的处理样本: {has_ctrl_train.sum()}/{len(treat_all)}", flush=True)

# ---------- 训练集划分 ----------
train_treat = np.where(train_mask & meta['role'].eq('treatment').values)[0]
train_ctrl = np.where(train_mask & meta['role'].eq('control').values)[0]
train_idx = np.concatenate([train_treat, train_ctrl])
print(f"[训练集] 处理 {len(train_treat)} + 对照 {len(train_ctrl)} = {len(train_idx)}", flush=True)

y_train = np.where(mask.astype(bool), y_log2, 0.0)[train_idx].astype(np.float32)
m_train = mask[train_idx].astype(np.float32)

# ---------- Δ 目标 + 低秩基 U（仅训练集处理样本，train-only）----------
treat_pos = np.array([pos_of[s] for s in train_treat])
delta_train = np.full((len(treat_pos), P), np.nan, dtype=np.float32)
for i, tp in enumerate(treat_pos):
    if has_ctrl_train[tp]:
        delta_train[i] = tr_y_nan[train_treat[i]] - ctrl_train_only[tp]
delta_filled = np.where(np.isfinite(delta_train), delta_train, 0.0).astype(np.float32)
delta_mask = np.isfinite(delta_train).astype(np.float32)
print(f"[Δ] 有效 Δ 值 {delta_mask.sum()/1e6:.2f}M / {delta_filled.shape}", flush=True)

from sklearn.decomposition import TruncatedSVD
print("[低秩基] 训练集 Δ SVD ...", flush=True)
svd = TruncatedSVD(n_components=64, random_state=42)
svd.fit(delta_filled)
U_basis = svd.components_.T.astype(np.float32)  # (P, 64)
U_basis = U_basis / (np.linalg.norm(U_basis, axis=0, keepdims=True) + 1e-8)
evr = svd.explained_variance_ratio_.sum()
print(f"[低秩基] U {U_basis.shape}, Δ 方差解释 {evr*100:.1f}%", flush=True)
np.save(f"{DATA}/v50_response_basis.npy", U_basis)

# ---------- 特征预加载 ----------
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
    }

# ---------- 模型 ----------
import importlib.util
_spec = importlib.util.spec_from_file_location("m50", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v50.py")
_m50 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m50)
model = _m50.VCellModel(feats, P=P, response_basis=U_basis).to(DEV)
model.set_strain_avg()
print(f"[模型] 参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M (v5.0 control+delta低秩)", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)
mse_loss = nn.MSELoss(reduction='none')

def masked_mse(pred, target, m):
    return (mse_loss(pred, target) * m).sum() / m.sum().clamp(min=1.0)

def corr_loss(pred, target, m):
    v = m.bool()
    if v.sum() < 10:
        return torch.tensor(0.0, device=pred.device)
    p, t = pred[v], target[v]
    p = p - p.mean(); t = t - t.mean()
    denom = torch.sqrt((p ** 2).sum() * (t ** 2).sum())
    if denom.item() == 0:
        return torch.tensor(0.0, device=pred.device)
    return 1.0 - (p * t).sum() / denom

def fc_corr_loss(pred, yt, m, yc):
    yc = yc.to(pred.device)
    v = m.bool() & torch.isfinite(yc)
    if v.sum() < 10:
        return torch.tensor(0.0, device=pred.device)
    dp = pred[v] - yc[v]; dt = yt[v] - yc[v]
    dp = dp - dp.mean(); dt = dt - dt.mean()
    denom = torch.sqrt((dp ** 2).sum() * (dt ** 2).sum())
    if denom.item() == 0:
        return torch.tensor(0.0, device=pred.device)
    return 1.0 - (dp * dt).sum() / denom

def per_protein_corr_loss(pred, yt, m):
    cnt = m.sum(0)
    keep = cnt >= 3
    if keep.sum() < 10:
        return torch.tensor(0.0, device=pred.device)
    pm = (pred * m).sum(0) / cnt.clamp(min=1)
    tm = (yt * m).sum(0) / cnt.clamp(min=1)
    pc = (pred - pm) * m
    tc = (yt - tm) * m
    num = (pc * tc).sum(0)
    den = torch.sqrt((pc ** 2).sum(0) * (tc ** 2).sum(0)).clamp(min=1e-8)
    return 1.0 - (num / den)[keep].mean()

# 蛋白共表达主成分（train-only，用于 graph reg）
print("[预计算] 蛋白共表达主成分 ...", flush=True)
X_tr_full = np.where(mask[train_mask].astype(bool), y_log2[train_mask], np.nan)
gmean_tr = np.nanmean(X_tr_full, axis=0)
Xc_graph = np.where(mask[train_mask].astype(bool), y_log2[train_mask] - gmean_tr, 0.0)
svd_g = TruncatedSVD(n_components=64, random_state=42)
svd_g.fit(Xc_graph)
V_graph = (svd_g.components_.T / (np.linalg.norm(svd_g.components_.T, axis=0, keepdims=True) + 1e-8)).astype(np.float32)
V_graph_t = torch.tensor(V_graph, device=DEV)

def graph_reg_loss(pred, m, V):
    cnt = m.sum(0).clamp(min=1)
    pm = (pred * m).sum(0) / cnt
    pc = (pred - pm) * m
    proj = pc @ V
    recon = proj @ V.T
    return ((pc - recon) ** 2 * m).sum() / m.sum()

# ---------- 评估 ----------
def evaluate(ep):
    model.eval()
    print(f"\n[Epoch {ep}] {'场景':<20}{'样本':>5}{'RMSE':>7}{'GlobalR2':>9}{'蛋白R2中位':>10}{'FC PCC':>8}", flush=True)
    for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
        idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
        if len(idx) == 0:
            continue
        x = make_x(idx)
        with torch.no_grad():
            pred = model(x).cpu().numpy()
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
        # 评估口径 FC：用全量对照（官方 M1 用真实对照）
        yc = ctrl_all[[pos_of[s] for s in idx]]
        fc_ok = np.isfinite(yc) & m & np.isfinite(pred)
        d_pred = (pred - yc)[fc_ok]; d_true = (yt - yc)[fc_ok]
        fc = float(np.corrcoef(d_pred, d_true)[0, 1]) if len(d_pred) > 10 else float('nan')
        print(f"{scene:<20}{len(idx):>5}{rmse:>7.3f}{g2:>9.3f}{p2:>10.3f}{fc:>8.3f}", flush=True)

# ---------- 评估用全量对照（观测值，官方 M1 口径；训练监督仍 train-only）----------
ctrl_all = np.full((len(treat_all), P), np.nan, dtype=np.float32)
print("[预计算] 全量对照（评估口径）...", flush=True)
ctrl_all_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_all_key = (meta.iloc[ctrl_all_idx]['data_source'].astype(str) + '|' + meta.iloc[ctrl_all_idx]['instrument'].astype(str) + '|'
                + meta.iloc[ctrl_all_idx]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_all_idx]['Strains'].astype(str) + '|'
                + meta.iloc[ctrl_all_idx]['Medium'].astype(str) + '|' + meta.iloc[ctrl_all_idx]['Temperature'].astype(str) + '|'
                + meta.iloc[ctrl_all_idx]['pert_time'].astype(str)).values
ctrl_all_lookup = {}
for k, pos in zip(ctrl_all_key, ctrl_all_idx):
    ctrl_all_lookup.setdefault(k, []).append(pos)
for i, sid in enumerate(treat_all):
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    rows = ctrl_all_lookup.get(k, [])
    if rows:
        cvals = tr_y_nan[rows]; cm = mask[rows] > 0
        with np.errstate(invalid='ignore'):
            ctrl_all[i] = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)

# ---------- 组件级残差监督目标（gpt2 步骤13 / P1：LOO μ_ctx / μ_drug，train-only）----------
print("[组件监督] LOO μ_ctx / μ_drug ...", flush=True)
chem_of_tr = meta['perturbation_no_concentration'].values[train_treat]
ctx_of_tr = (meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|'
             + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)).values[train_treat]
mu_ctx = np.full((len(treat_pos), P), np.nan)
mu_drug = np.full((len(treat_pos), P), np.nan)
for key, members in pd.Series(np.arange(len(treat_pos)), index=ctx_of_tr).groupby(level=0):
    members = members.values
    if len(members) > 1:
        s = np.nansum(delta_train[members], axis=0)
        n = np.sum(np.isfinite(delta_train[members]), axis=0)
        for m in members:
            msk = np.isfinite(delta_train[m])
            mu_ctx[m] = np.where(n > 0, (s - np.where(msk, delta_train[m], 0)) / np.maximum(n - msk.astype(float), 1), np.nan)
for key, members in pd.Series(np.arange(len(treat_pos)), index=chem_of_tr).groupby(level=0):
    members = members.values
    if len(members) > 1:
        s = np.nansum(delta_train[members], axis=0)
        n = np.sum(np.isfinite(delta_train[members]), axis=0)
        for m in members:
            msk = np.isfinite(delta_train[m])
            mu_drug[m] = np.where(n > 0, (s - np.where(msk, delta_train[m], 0)) / np.maximum(n - msk.astype(float), 1), np.nan)
resid_ctx = np.where(np.isfinite(delta_train) & np.isfinite(mu_ctx), delta_train - mu_ctx, 0.0).astype(np.float32)
resid_drug = np.where(np.isfinite(delta_train) & np.isfinite(mu_drug), delta_train - mu_drug, 0.0).astype(np.float32)
mask_resid_ctx = (np.isfinite(delta_train) & np.isfinite(mu_ctx) & (mu_ctx != 0)).astype(np.float32)
mask_resid_drug = (np.isfinite(delta_train) & np.isfinite(mu_drug) & (mu_drug != 0)).astype(np.float32)
print(f"[组件监督] 有效 ctx 残差 {mask_resid_ctx.sum()/1e6:.2f}M / drug 残差 {mask_resid_drug.sum()/1e6:.2f}M", flush=True)

# ---------- 训练张量 ----------
y_train_t = torch.tensor(y_train, device=DEV)
m_train_t = torch.tensor(m_train, device=DEV)
delta_train_t = torch.tensor(delta_filled, device=DEV)
delta_mask_t = torch.tensor(delta_mask, device=DEV)
y_ctrl_train_t = torch.tensor(ctrl_train_only[treat_pos], device=DEV)  # train-only 对照
y_ctrl_train_t = torch.nan_to_num(y_ctrl_train_t, nan=0.0)  # ★ 无对照蛋白列 NaN→0（配合 ctrl_mask）
has_ctrl_t = torch.tensor(has_ctrl_train[treat_pos], device=DEV)

treat_in_train = np.isin(train_idx, train_treat)

# control 分支监督目标：
#   - 对照样本：自身真值 y
#   - 处理样本：matched control（train-only）
ctrl_target = y_train_t.clone()
ctrl_mask = m_train_t.clone()
ctrl_target[~treat_in_train] = y_train_t[~treat_in_train]      # 对照行
ctrl_mask[~treat_in_train] = m_train_t[~treat_in_train]
# 处理行：目标 = matched control（仅当有对照）
ctrl_target[treat_in_train] = torch.where(has_ctrl_t[:, None].expand(-1, P), y_ctrl_train_t, 0.0)
ctrl_mask[treat_in_train] = torch.where(has_ctrl_t[:, None].expand(-1, P),
                                        torch.tensor(mask[train_treat], device=DEV), 0.0)
# Δ 目标：处理行
delta_target_full = torch.zeros_like(y_train_t)
delta_mask_full = torch.zeros_like(m_train_t)
delta_target_full[treat_in_train] = delta_train_t
delta_mask_full[treat_in_train] = delta_mask_t

# 组件监督张量（处理行索引）
resid_ctx_t = torch.tensor(resid_ctx, device=DEV)
resid_drug_t = torch.tensor(resid_drug, device=DEV)
mask_resid_ctx_t = torch.tensor(mask_resid_ctx, device=DEV)
mask_resid_drug_t = torch.tensor(mask_resid_drug, device=DEV)

# ---------- 三阶段训练（gpt2 阶段 A/B/C）----------
# 阶段 A (1-40)：control 预训练，只优化 ctrl 分支
# 阶段 B (41-100)：冻结 control，训练 response（delta/fc）
# 阶段 C (101-140)：联合微调（低 lr）
EPOCHS_A, EPOCHS_B, EPOCHS_C = 40, 60, 40
EPOCHS = EPOCHS_A + EPOCHS_B + EPOCHS_C
BATCH = 128

def freeze_ctrl(model, freeze):
    """冻结/解冻 control 分支参数"""
    for name, p in model.named_parameters():
        if 'ctrl_' in name or name == 'ctrl_bias':
            p.requires_grad = not freeze

def freeze_resp(model, freeze):
    for name, p in model.named_parameters():
        if 'resp_' in name or 'resid' in name or 'gate_c' in name or 'gate_s' in name \
           or 'w_esm' in name or 'gate_esm' in name or 'go_effect' in name or 'gate_go' in name \
           or 'chem_emb' in name or 'morgan' in name:
            p.requires_grad = not freeze

best_score = float('inf')
for ep in range(1, EPOCHS + 1):
    phase = 'A' if ep <= EPOCHS_A else ('B' if ep <= EPOCHS_A + EPOCHS_B else 'C')
    if phase == 'A':
        freeze_resp(model, True); freeze_ctrl(model, False)
    elif phase == 'B':
        freeze_ctrl(model, True); freeze_resp(model, False)
    else:
        freeze_ctrl(model, False); freeze_resp(model, False)
        # 阶段 C 降低学习率
        if ep == EPOCHS_A + EPOCHS_B + 1:
            for g in optimizer.param_groups:
                g['lr'] = 3e-4

    model.train()
    perm = np.random.permutation(len(train_idx))
    total_loss = 0.0; n_batch = 0
    pbar = tqdm(range(0, len(perm), BATCH), desc=f"Ep {ep}/{EPOCHS}[{phase}]", leave=False, ncols=100, file=sys.stdout)
    for i in pbar:
        b = perm[i:i + BATCH]
        x = make_x(train_idx[b])
        y_ctrl_pred = model.ctrl_predict(x)
        delta_pred = model.delta_predict(x)
        y_pred = y_ctrl_pred + delta_pred
        yt, m = y_train_t[b], m_train_t[b]

        loss_y = masked_mse(y_pred, yt, m)
        loss_ctrl = masked_mse(y_ctrl_pred, ctrl_target[b], ctrl_mask[b])
        # delta 监督：仅处理样本
        is_treat = treat_in_train[b]
        if is_treat.any():
            db = np.where(is_treat)[0]
            loss_delta = masked_mse(delta_pred[db], delta_target_full[b[db]], delta_mask_full[b[db]])
        else:
            loss_delta = torch.tensor(0.0, device=DEV)
        # FC corr（处理样本，train-only 对照）
        loss_fc = torch.tensor(0.0, device=DEV)
        # ★ 组件级残差监督（gpt2 步骤13）：delta_pred 对齐 LOO μ 残差
        loss_ctx = torch.tensor(0.0, device=DEV)
        loss_drug = torch.tensor(0.0, device=DEV)
        if is_treat.any():
            pos = np.where(is_treat)[0]
            yc = y_ctrl_train_t[b[pos]]
            fc_ok = torch.isfinite(yc).any(dim=1)
            if fc_ok.any():
                loss_fc = fc_corr_loss(y_pred[pos][fc_ok], yt[pos][fc_ok], m[pos][fc_ok], yc[fc_ok])
            # 组件监督：delta_pred vs LOO 残差（仅阶段 B/C 启用，代码内按 phase 加权）
            dpi = b[pos]
            loss_ctx = corr_loss(delta_pred[pos], resid_ctx_t[dpi], mask_resid_ctx_t[dpi])
            loss_drug = corr_loss(delta_pred[pos], resid_drug_t[dpi], mask_resid_drug_t[dpi])

        loss_prot = per_protein_corr_loss(y_pred, yt, m)
        loss_graph = graph_reg_loss(y_pred, m, V_graph_t)

        if phase == 'A':
            loss = loss_ctrl  # control 预训练
        elif phase == 'B':
            loss = (0.3 * loss_y + 0.35 * loss_delta + 0.15 * loss_fc
                    + 0.1 * loss_ctx + 0.1 * loss_drug
                    + 0.1 * loss_prot + 0.05 * loss_graph)
        else:
            loss = (0.3 * loss_y + 0.15 * loss_ctrl + 0.35 * loss_delta + 0.15 * loss_fc
                    + 0.1 * loss_ctx + 0.1 * loss_drug
                    + 0.1 * loss_prot + 0.05 * loss_graph)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item(); n_batch += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", y=f"{loss_y.item():.4f}", d=f"{loss_delta.item():.4f}")
    avg_loss = total_loss / max(n_batch, 1)
    scheduler.step(avg_loss)
    if ep % 10 == 0 or ep == EPOCHS:
        evaluate(ep)
        torch.save(model.state_dict(), f"{DATA}/model_v50_{_SEED}.pt")
    if avg_loss < best_score:
        best_score = avg_loss
        torch.save(model.state_dict(), f"{DATA}/model_v50_{_SEED}_best.pt")
    print(f"  avg_loss={avg_loss:.4f} [phase {phase}]", flush=True)

print("\n训练完成", flush=True)
