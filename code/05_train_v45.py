# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 05 训练 v4.5（GO 通路重激活微调）
诊断发现 v37 中 gate_go=0.21（只激活 21%）、go_effect 权重 std=0.042（被压扁）。
假设：GO 通路注意力被欠训练——初始化 0.1 导致梯度不足，模型倾向走 proj 主路径。
实验：加载 v37 权重，冻结主路径，只微调 GO 相关参数（go_effect/gate_go），
      提高初始 gate_go，看 GO 通路能否重新激活并带来提升。
这是单变量实验：不改变主结构，只让已有组件发挥充分。
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
ctx_prior_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)

# ---------- matched control 预计算 ----------
ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = (meta.iloc[ctrl_idx]['data_source'].astype(str) + '|' + meta.iloc[ctrl_idx]['instrument'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_idx]['Strains'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Medium'].astype(str) + '|' + meta.iloc[ctrl_idx]['Temperature'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['pert_time'].astype(str)).values
ctrl_lookup = {}
for k, pos in zip(ctrl_key, ctrl_idx):
    ctrl_lookup.setdefault(k, []).append(pos)
tr_y_nan = np.where(mask.astype(bool), y_log2, np.nan)

def matched_control_mean(sid):
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
ctrl_all = np.full((len(treat_all), P), np.nan, dtype=np.float32)
for i, sid in enumerate(treat_all):
    cm = matched_control_mean(sid)
    if cm is not None:
        ctrl_all[i] = cm
pos_of = {sid: i for i, sid in enumerate(treat_all)}

train_treat = np.where(train_mask & meta['role'].eq('treatment').values)[0]
train_ctrl = np.where(train_mask & meta['role'].eq('control').values)[0]
train_idx = np.concatenate([train_treat, train_ctrl])
y_train = np.where(mask.astype(bool), y_log2, 0.0)[train_idx].astype(np.float32)
m_train = mask[train_idx].astype(np.float32)

# 组件监督目标（LOO）
delta_treat = np.full((len(treat_all), P), np.nan)
for i in range(len(treat_all)):
    delta_treat[i] = tr_y_nan[treat_all[i]] - ctrl_all[i]
chem_of = meta['perturbation_no_concentration'].values[treat_all]
ctx_of = (meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|'
          + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)).values[treat_all]
mu_ctx = np.full((len(treat_all), P), np.nan)
mu_drug = np.full((len(treat_all), P), np.nan)
for key, members in pd.Series(np.arange(len(treat_all)), index=ctx_of).groupby(level=0):
    members = members.values
    if len(members) > 1:
        s = np.nansum(delta_treat[members], axis=0)
        n = np.sum(~np.isnan(delta_treat[members]), axis=0)
        for m in members:
            msk = ~np.isnan(delta_treat[m])
            mu_ctx[m] = np.where(n > 0, (s - np.where(msk, delta_treat[m], 0)) / np.maximum(n - msk.astype(float), 1), np.nan)
for key, members in pd.Series(np.arange(len(treat_all)), index=chem_of).groupby(level=0):
    members = members.values
    if len(members) > 1:
        s = np.nansum(delta_treat[members], axis=0)
        n = np.sum(~np.isnan(delta_treat[members]), axis=0)
        for m in members:
            msk = ~np.isnan(delta_treat[m])
            mu_drug[m] = np.where(n > 0, (s - np.where(msk, delta_treat[m], 0)) / np.maximum(n - msk.astype(float), 1), np.nan)
treat_pos = np.array([pos_of[s] for s in train_treat])
resid_ctx = delta_treat[treat_pos] - mu_ctx[treat_pos]
resid_drug = delta_treat[treat_pos] - mu_drug[treat_pos]
resid_ctx = np.where(np.isnan(resid_ctx), 0.0, resid_ctx).astype(np.float32)
resid_drug = np.where(np.isnan(resid_drug), 0.0, resid_drug).astype(np.float32)
mask_resid = np.isfinite(delta_treat[treat_pos]) & np.isfinite(mu_ctx[treat_pos]) & (mu_ctx[treat_pos] != 0)
mask_resid_drug = np.isfinite(delta_treat[treat_pos]) & np.isfinite(mu_drug[treat_pos]) & (mu_drug[treat_pos] != 0)

# ---------- 预加载特征 ----------
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

# ---------- 模型（加载 v37 权重）----------
import importlib.util
_spec = importlib.util.spec_from_file_location("m37", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v37.py")
_m37 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m37)
model = _m37.VCellModel(feats, P=P).to(DEV)
sd = torch.load(f"{DATA}/model_v37_42_best.pt", map_location=DEV, weights_only=True)
model.load_state_dict(sd)
model.set_strain_avg()
print(f"[模型] 加载 v37 权重完成，gate_go 初始 = {torch.sigmoid(model.gate_go).item():.3f}", flush=True)

# ★ 冻结除 GO 相关外的所有参数（单变量微调）
FREEZE_KEYS = ['go_effect', 'gate_go', 'gate_esm', 'w_esm']
trainable = []
for name, p in model.named_parameters():
    if any(k in name for k in FREEZE_KEYS):
        p.requires_grad = True
        trainable.append(name)
    else:
        p.requires_grad = False
print(f"[冻结] 可训练参数 {len(trainable)} 个: {trainable}", flush=True)

# 提高 gate_go 初始值，给 GO 足够梯度空间
with torch.no_grad():
    model.gate_go.fill_(1.5)  # sigmoid(1.5)≈0.82
print(f"[初始化] gate_go 重置为 1.5 (sigmoid≈{torch.sigmoid(model.gate_go).item():.3f})", flush=True)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
mse_loss = nn.MSELoss(reduction='none')

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
    corr = num / den
    return 1.0 - corr[keep].mean()

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
        yc = ctrl_all[[pos_of[s] for s in idx]]
        fc_ok = np.isfinite(yc) & m & np.isfinite(pred)
        d_pred = (pred - yc)[fc_ok]; d_true = (yt - yc)[fc_ok]
        fc = float(np.corrcoef(d_pred, d_true)[0, 1]) if len(d_pred) > 10 else float('nan')
        print(f"{scene:<20}{len(idx):>5}{rmse:>7.3f}{g2:>9.3f}{p2:>10.3f}{fc:>8.3f}", flush=True)

# ---------- 训练 ----------
y_train_t = torch.tensor(y_train, device=DEV)
m_train_t = torch.tensor(m_train, device=DEV)
y_ctrl_all_t = torch.tensor(ctrl_all[treat_pos], device=DEV)
resid_ctx_t = torch.tensor(resid_ctx, device=DEV)
resid_drug_t = torch.tensor(resid_drug, device=DEV)
mask_resid_t = torch.tensor(mask_resid, device=DEV)
mask_resid_drug_t = torch.tensor(mask_resid_drug, device=DEV)
treat_in_train = np.isin(train_idx, train_treat)

from sklearn.decomposition import TruncatedSVD
X_tr_full = np.where(mask[train_mask].astype(bool), y_log2[train_mask], np.nan)
gmean_tr = np.nanmean(X_tr_full, axis=0)
Xc_graph = np.where(mask[train_mask].astype(bool), y_log2[train_mask] - gmean_tr, 0.0)
svd = TruncatedSVD(n_components=64, random_state=42)
svd.fit(Xc_graph)
V_graph = svd.components_.T.astype(np.float32)
V_graph = V_graph / (np.linalg.norm(V_graph, axis=0, keepdims=True) + 1e-8)
V_graph_t = torch.tensor(V_graph, device=DEV)

EPOCHS, BATCH = 60, 128
best_score = float('inf')
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = np.random.permutation(len(train_idx))
    total_loss = 0.0; n_batch = 0
    pbar = tqdm(range(0, len(perm), BATCH), desc=f"Ep {ep}/{EPOCHS}", leave=False, ncols=100, file=sys.stdout)
    for i in pbar:
        b = perm[i:i + BATCH]
        x = make_x(train_idx[b])
        pred = model(x)
        yt, m = y_train_t[b], m_train_t[b]
        loss_mse = (mse_loss(pred, yt) * m).sum() / m.sum()
        loss = 0.5 * loss_mse
        is_treat = treat_in_train[b]
        if is_treat.any():
            pos = np.where(is_treat)[0]
            yc = y_ctrl_all_t[b[pos]]
            fc_ok = torch.isfinite(yc).any(dim=1)
            if fc_ok.any():
                loss_fc_val = fc_corr_loss(pred[pos][fc_ok], yt[pos][fc_ok], m[pos][fc_ok], yc[fc_ok])
                loss += 0.25 * loss_fc_val
        if is_treat.any():
            pos = np.where(is_treat)[0]
            yB, yS, yC, yT = model.components({k: (v[pos] if k in ('ctx_prior', 'chem_morgan') else [t[pos] for t in v])
                                                for k, v in x.items()})
            loss_ctx_val = corr_loss(yC, resid_ctx_t[b[pos]], mask_resid_t[b[pos]])
            loss_drug_val = corr_loss(yT, resid_drug_t[b[pos]], mask_resid_drug_t[b[pos]])
            loss += 0.25 * loss_ctx_val + 0.25 * loss_drug_val
        loss_prot = per_protein_corr_loss(pred, yt, m)
        loss_graph = graph_reg_loss(pred, m, V_graph_t)
        loss += 0.1 * loss_prot + 0.05 * loss_graph
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item(); n_batch += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = total_loss / max(n_batch, 1)
    scheduler.step(avg_loss)
    if ep % 10 == 0 or ep == EPOCHS:
        evaluate(ep)
        torch.save(model.state_dict(), f"{DATA}/model_v45_{_SEED}.pt")
    if avg_loss < best_score:
        best_score = avg_loss
        torch.save(model.state_dict(), f"{DATA}/model_v45_{_SEED}_best.pt")
    print(f"  avg_loss={avg_loss:.4f}  gate_go={torch.sigmoid(model.gate_go).item():.3f}", flush=True)

print("\n训练完成", flush=True)
