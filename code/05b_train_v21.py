# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 05b 训练 v2.1（对照监督 + 组件级监督 + 后处理校准对比）
增量 vs v2.0：
  1) 对照样本加入训练（B 分支学基础水平）
  2) 组件级监督：yC ↔ Δ_true−μ_ctx(LOO) / yT ↔ Δ_true−μ_drug(LOO)（corr 形式，复用模块头投影）
  3) 评估时对比「原始预测」vs「后处理 control 校准」
"""
import os, pickle, numpy as np, pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[设备] {DEV}", flush=True)

meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy").astype(np.float32)
mask = np.load(f"{DATA}/mask.npy").astype(np.float32)
with open(f"{DATA}/feats.pkl", 'rb') as f:
    feats = pickle.load(f)
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values

# ---------- matched control 预计算（全局一次）----------
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
print("[预计算] matched control ...", flush=True)
ctrl_all = np.full((len(treat_all), P), np.nan, dtype=np.float32)
for i, sid in enumerate(treat_all):
    cm = matched_control_mean(sid)
    if cm is not None:
        ctrl_all[i] = cm
pos_of = {sid: i for i, sid in enumerate(treat_all)}
print("[预计算] 完成", flush=True)

# ---------- 训练集（处理 + 对照）----------
train_treat = np.where(train_mask & meta['role'].eq('treatment').values)[0]
train_ctrl = np.where(train_mask & meta['role'].eq('control').values)[0]
train_idx = np.concatenate([train_treat, train_ctrl])
print(f"[训练集] 处理 {len(train_treat)} + 对照 {len(train_ctrl)} = {len(train_idx)}", flush=True)

y_train = np.where(mask.astype(bool), y_log2, 0.0)[train_idx].astype(np.float32)
m_train = mask[train_idx].astype(np.float32)

# ---------- 组件监督目标（LOO 计算 Δ 均值）----------
delta_treat = np.full((len(treat_all), P), np.nan)
for i in range(len(treat_all)):
    delta_treat[i] = tr_y_nan[treat_all[i]] - ctrl_all[i]
chem_of = meta['perturbation_no_concentration'].values[treat_all]
ctx_of = (meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|'
          + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)).values[treat_all]
print("[组件监督] 计算 LOO μ_ctx / μ_drug ...", flush=True)
mu_ctx = np.full((len(treat_all), P), np.nan)
mu_drug = np.full((len(treat_all), P), np.nan)
# 按组求和 + 计数（非 NaN 位置）
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
print("[组件监督] 完成", flush=True)

# 训练处理样本的组件目标
treat_pos = np.array([pos_of[s] for s in train_treat])
resid_ctx = delta_treat[treat_pos] - mu_ctx[treat_pos]     # (n_treat, P) 目标 for C
resid_drug = delta_treat[treat_pos] - mu_drug[treat_pos]   # (n_treat, P) 目标 for T
resid_ctx = np.where(np.isnan(resid_ctx), 0.0, resid_ctx).astype(np.float32)
resid_drug = np.where(np.isnan(resid_drug), 0.0, resid_drug).astype(np.float32)
mask_resid = np.isfinite(delta_treat[treat_pos]) & np.isfinite(mu_ctx[treat_pos]) & (mu_ctx[treat_pos] != 0)
mask_resid_drug = np.isfinite(delta_treat[treat_pos]) & np.isfinite(mu_drug[treat_pos]) & (mu_drug[treat_pos] != 0)

# ---------- 特征张量 ----------
def make_x(idx):
    return {
        'bio': [torch.from_numpy(feats['strain_id'][idx]), torch.from_numpy(feats['chem_id'][idx]),
                torch.from_numpy(feats['chem_hash'][idx]), torch.from_numpy(feats['medium_onehot'][idx]),
                torch.from_numpy(feats['temp_norm'][idx]), torch.from_numpy(feats['time_feat'][idx]),
                torch.from_numpy(feats['sm_id'][idx]), torch.from_numpy(feats['ct_id'][idx])],
        'ctx': [torch.from_numpy(feats['src_id'][idx]), torch.from_numpy(feats['ins_id'][idx]),
                torch.from_numpy(feats['plt_id'][idx])],
        'seen': [torch.from_numpy(feats['chem_seen'][idx]), torch.from_numpy(feats['strain_seen'][idx])],
    }

# ---------- 模型 ----------
import importlib.util
_spec = importlib.util.spec_from_file_location("model04", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model.py")
_m04 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m04)
model = _m04.VCellModel(feats, P=P).to(DEV)
print(f"[模型] 参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
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

# ---------- 评估（含后处理校准对比）----------
def evaluate(ep, use_calib=False):
    model.eval()
    tag = "校准后" if use_calib else "原始"
    print(f"\n[Epoch {ep} | {tag}] {'场景':<16}{'样本':>5}{'RMSE':>7}{'GlobalR2':>9}{'蛋白R2中位':>10}{'FC PCC':>8}", flush=True)
    for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
        idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
        if len(idx) == 0:
            continue
        yc = ctrl_all[[pos_of[s] for s in idx]]
        x = make_x(idx)
        with torch.no_grad():
            pred = model({k: [t.to(DEV) for t in v] for k, v in x.items()}).cpu().numpy()
        if use_calib:
            gmean = feats['gmean']
            ok = np.isfinite(yc)
            pred = np.where(ok, pred - yc + gmean, pred)   # control 偏移校准
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
        fc_ok = np.isfinite(yc) & m
        if use_calib:
            fc_ok &= np.isfinite(pred)
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

EPOCHS, BATCH = 150, 256
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = np.random.permutation(len(train_idx))
    pbar = tqdm(range(0, len(perm), BATCH), desc=f"Ep {ep}/{EPOCHS}", leave=False, ncols=100)
    for i in pbar:
        b = perm[i:i + BATCH]
        x = make_x(train_idx[b])
        x = {k: [t.to(DEV) for t in v] for k, v in x.items()}
        pred = model(x)
        yt, m = y_train_t[b], m_train_t[b]
        loss_mse = (mse_loss(pred, yt) * m).sum() / m.sum()
        loss = loss_mse
        # FC corr（处理样本）
        is_treat = treat_in_train[b]
        if is_treat.any():
            pos = np.where(is_treat)[0]
            yc = y_ctrl_all_t[b[pos]]
            fc_ok = torch.isfinite(yc).any(dim=1)
            if fc_ok.any():
                loss += 0.4 * fc_corr_loss(pred[pos][fc_ok], yt[pos][fc_ok], m[pos][fc_ok], yc[fc_ok])
        # 组件级监督（处理样本的 C/T 分支）
        if is_treat.any():
            pos = np.where(is_treat)[0]
            yB, yS, yC, yT = model.components({k: [t[pos] for t in v] for k, v in x.items()})
            loss += 0.15 * corr_loss(yC, resid_ctx_t[b[pos]], mask_resid_t[b[pos]])
            loss += 0.15 * corr_loss(yT, resid_drug_t[b[pos]], mask_resid_drug_t[b[pos]])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}", mse=f"{loss_mse.item():.4f}")
    scheduler.step()
    if ep % 10 == 0 or ep == EPOCHS:
        evaluate(ep, use_calib=False)
        evaluate(ep, use_calib=True)
        torch.save(model.state_dict(), f"{DATA}/model_v21.pt")
print("\n训练完成", flush=True)
