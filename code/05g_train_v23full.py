# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 05d 训练 v2.3（架构修正版）
修正：gate 随机化 + latent 128 + 去 hash；组件监督 + 对照监督 + 解耦正则保留
GPU 利用率优化：特征预取 GPU、postfix 每 5 batch 才同步 loss
"""
import os, sys, pickle, numpy as np, pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_DECOUPLE = '--no_decouple' not in sys.argv
USE_DROP = '--no_drop' not in sys.argv
print(f"[设备] {DEV} | decouple={USE_DECOUPLE} | ent_drop={USE_DROP}", flush=True)

meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy").astype(np.float32)
mask = np.load(f"{DATA}/mask.npy").astype(np.float32)
with open(f"{DATA}/feats.pkl", 'rb') as f:
    feats = pickle.load(f)
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values

# ---------- matched control（全局一次）----------
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

# ---------- 训练集 ----------
train_treat = np.where(train_mask & meta['role'].eq('treatment').values)[0]
train_ctrl = np.where(train_mask & meta['role'].eq('control').values)[0]
train_idx = np.concatenate([train_treat, train_ctrl])
y_train = np.where(mask.astype(bool), y_log2, 0.0)[train_idx].astype(np.float32)
m_train = mask[train_idx].astype(np.float32)
treat_in_train = np.isin(train_idx, train_treat)

# ---------- 组件监督目标（LOO）----------
delta_treat = np.full((len(treat_all), P), np.nan)
for i in range(len(treat_all)):
    delta_treat[i] = tr_y_nan[treat_all[i]] - ctrl_all[i]
chem_of = meta['perturbation_no_concentration'].values[treat_all]
ctx_of = (meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|'
          + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)).values[treat_all]
print("[组件监督] LOO μ_ctx/μ_drug ...", flush=True)
mu_ctx = np.full((len(treat_all), P), np.nan); mu_drug = np.full((len(treat_all), P), np.nan)
for key, members in pd.Series(np.arange(len(treat_all)), index=ctx_of).groupby(level=0):
    members = members.values
    if len(members) > 1:
        s = np.nansum(delta_treat[members], axis=0); n = np.sum(~np.isnan(delta_treat[members]), axis=0)
        for m in members:
            msk = ~np.isnan(delta_treat[m])
            mu_ctx[m] = np.where(n > 0, (s - np.where(msk, delta_treat[m], 0)) / np.maximum(n - msk.astype(float), 1), np.nan)
for key, members in pd.Series(np.arange(len(treat_all)), index=chem_of).groupby(level=0):
    members = members.values
    if len(members) > 1:
        s = np.nansum(delta_treat[members], axis=0); n = np.sum(~np.isnan(delta_treat[members]), axis=0)
        for m in members:
            msk = ~np.isnan(delta_treat[m])
            mu_drug[m] = np.where(n > 0, (s - np.where(msk, delta_treat[m], 0)) / np.maximum(n - msk.astype(float), 1), np.nan)
treat_pos = np.array([pos_of[s] for s in train_treat])
resid_ctx = np.where(np.isnan(delta_treat[treat_pos] - mu_ctx[treat_pos]), 0.0, delta_treat[treat_pos] - mu_ctx[treat_pos]).astype(np.float32)
resid_drug = np.where(np.isnan(delta_treat[treat_pos] - mu_drug[treat_pos]), 0.0, delta_treat[treat_pos] - mu_drug[treat_pos]).astype(np.float32)
mask_resid = np.isfinite(delta_treat[treat_pos]) & np.isfinite(mu_ctx[treat_pos]) & (mu_ctx[treat_pos] != 0)
mask_resid_drug = np.isfinite(delta_treat[treat_pos]) & np.isfinite(mu_drug[treat_pos]) & (mu_drug[treat_pos] != 0)
print("[组件监督] 完成", flush=True)

# ---------- 特征（全量预取 GPU）----------
BIO_KEYS = ['strain_id', 'chem_id', 'medium_onehot', 'temp_norm', 'time_feat', 'sm_id', 'ct_id']
CTX_KEYS = ['src_id', 'ins_id', 'plt_id']
bio_all = [torch.from_numpy(feats[k]).to(DEV) for k in BIO_KEYS]
ctx_all = [torch.from_numpy(feats[k]).to(DEV) for k in CTX_KEYS]
seen_all = [torch.from_numpy(feats['chem_seen']).to(DEV), torch.from_numpy(feats['strain_seen']).to(DEV)]

def make_x(idx):
    return {
        'bio': [t[idx] for t in bio_all],
        'ctx': [t[idx] for t in ctx_all],
        'seen': [t[idx] for t in seen_all],
    }

# ---------- 模型 ----------
import importlib.util
_spec = importlib.util.spec_from_file_location("model04", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model.py")
_m04 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m04)
model = _m04.VCellModel(feats, P=P, latent=128, drop_ent=0.15 if USE_DROP else 0.0).to(DEV)
print(f"[模型] 参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M | latent=128", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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

# ---------- 评估 ----------
def evaluate(ep):
    model.eval()
    print(f"\n[Epoch {ep}] {'场景':<16}{'样本':>5}{'RMSE':>7}{'GlobalR2':>9}{'蛋白R2中位':>10}{'FC PCC':>8}", flush=True)
    res = {}
    for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
        idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
        if len(idx) == 0:
            continue
        yc = ctrl_all[[pos_of[s] for s in idx]]
        x = make_x(idx)
        with torch.no_grad():
            pred = model({k: [t.to(DEV) for t in v] for k, v in x.items()}).cpu().numpy()
        yt, m = y_log2[idx], mask[idx].astype(bool)
        valid = m & np.isfinite(pred)
        rmse = float(np.sqrt(((yt[m] - pred[m]) ** 2).mean()))
        a, b = yt[valid], pred[valid]
        g2 = 1 - ((a - b) ** 2).sum() / max(((a - a.mean()) ** 2).sum(), 1e-12)
        cnt = valid.sum(0); keep = cnt >= 3; n = np.maximum(cnt.astype(float), 1)
        ytc = np.where(valid, yt, 0.0); ypc = np.where(valid, pred, 0.0)
        mt = ytc.sum(0) / n
        ss_tot = (((ytc - mt) ** 2) * valid).sum(0); ss_res = (((ytc - ypc) ** 2) * valid).sum(0)
        p2 = float(np.median(1 - ss_res / np.maximum(ss_tot, 1e-12)))
        fc_ok = np.isfinite(yc) & m
        d_pred = (pred - yc)[fc_ok]; d_true = (yt - yc)[fc_ok]
        fc = float(np.corrcoef(d_pred, d_true)[0, 1]) if len(d_pred) > 10 else float('nan')
        print(f"{scene:<20}{len(idx):>5}{rmse:>7.3f}{g2:>9.3f}{p2:>10.3f}{fc:>8.3f}", flush=True)
        res[scene] = (rmse, g2, p2, fc)
    return res

# ---------- 训练 ----------
y_train_t = torch.tensor(y_train, device=DEV)
m_train_t = torch.tensor(m_train, device=DEV)
y_ctrl_t = torch.tensor(ctrl_all[treat_pos], device=DEV)
resid_ctx_t = torch.tensor(resid_ctx, device=DEV); resid_drug_t = torch.tensor(resid_drug, device=DEV)
mask_resid_t = torch.tensor(mask_resid, device=DEV); mask_resid_drug_t = torch.tensor(mask_resid_drug, device=DEV)

CTRL_P2 = {'val_chem_only': 0.836, 'val_strain_only': 0.726, 'val_both': 0.809, 'val_time': 0.751}
EPOCHS, BATCH = 150, 256
best_score = -9
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(train_idx), device=DEV)   # GPU 端打乱，省 CPU-GPU 往返
    pbar = tqdm(range(0, len(perm), BATCH), desc=f"Ep {ep}/{EPOCHS}", leave=False, ncols=100)
    for i in pbar:
        b = perm[i:i + BATCH]
        x = make_x(b)
        pred, yC, yT = model(x, with_components=True)
        yt, m = y_train_t[b], m_train_t[b]
        loss_mse = (mse_loss(pred, yt) * m).sum() / m.sum()
        loss = loss_mse
        is_treat = treat_in_train[b.cpu().numpy()]
        if is_treat.any():
            pos = torch.where(torch.from_numpy(is_treat))[0]
            yc = y_ctrl_t[b[pos]]
            fc_ok = torch.isfinite(yc).any(dim=1)
            if fc_ok.any():
                loss += 0.4 * fc_corr_loss(pred[pos][fc_ok], yt[pos][fc_ok], m[pos][fc_ok], yc[fc_ok])
            loss += 0.15 * corr_loss(yC[pos], resid_ctx_t[b[pos]], mask_resid_t[b[pos]])
            loss += 0.15 * corr_loss(yT[pos], resid_drug_t[b[pos]], mask_resid_drug_t[b[pos]])
            if USE_DECOUPLE and (i // BATCH) % 2 == 0:
                xp = {k: t[pos] for k, v in x.items() for t in [v] if isinstance(v, torch.Tensor)} if False else {k: [t[pos] for t in v] for k, v in x.items()}
                z, _, _, c, t, g_c, g_s = model(xp, ret_latent=True)
                zns = model.neutral_latent(xp, 'strain')
                znc = model.neutral_latent(xp, 'chem')
                loss += 0.1 * mse_loss(zns, (z - g_s * t).detach()).mean()
                loss += 0.1 * mse_loss(znc, (z - g_c * c).detach()).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if i // BATCH % 5 == 0:   # 每 5 batch 同步一次 loss，减 CPU-GPU 往返
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    scheduler.step()
    if ep % 10 == 0 or ep == EPOCHS:
        res = evaluate(ep)
        score = 0.4 * (res['val_both'][2] - CTRL_P2['val_both']) + 0.3 * (res['val_chem_only'][2] - CTRL_P2['val_chem_only']) + 0.3 * (res['val_time'][2] - CTRL_P2['val_time'])
        print(f"[score] {score:.4f} (best {best_score:.4f})", flush=True)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f"{DATA}/model_v23m_best.pt")
    torch.save(model.state_dict(), f"{DATA}/model_v23m_last.pt")
print("\n训练完成", flush=True)
