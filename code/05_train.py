# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 05 训练（v2 架构）
mask-aware MSE + FC corr loss，全参数训练，tqdm 全程显示
每 10 epoch 在 val 四场景评估（RMSE / GlobalR² / 蛋白R²中位 / FC PCC）
"""
import os, pickle, numpy as np, pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[设备] {DEV}: {torch.cuda.get_device_name(0) if DEV=='cuda' else 'CPU'}", flush=True)

meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy").astype(np.float32)
mask = np.load(f"{DATA}/mask.npy").astype(np.float32)
with open(f"{DATA}/feats.pkl", 'rb') as f:
    feats = pickle.load(f)
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values

# ---------- 预计算 matched control 均值（全局一次）----------
ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = (meta.iloc[ctrl_idx]['data_source'].astype(str) + '|' + meta.iloc[ctrl_idx]['instrument'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_idx]['Strains'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Medium'].astype(str) + '|' + meta.iloc[ctrl_idx]['Temperature'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['pert_time'].astype(str)).values
ctrl_lookup = {}
for k, pos in zip(ctrl_key, ctrl_idx):
    ctrl_lookup.setdefault(k, []).append(pos)

tr_y_nan = np.where(mask.astype(bool), y_log2, np.nan)   # 全局一次

def matched_control_mean(sid):
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    if k not in ctrl_lookup:
        return None
    rows = ctrl_lookup[k]
    cvals = tr_y_nan[rows]
    cm = mask[rows] > 0
    with np.errstate(invalid='ignore'):
        return np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)

def get_ctrl_for(idx_list):
    out = np.full((len(idx_list), P), np.nan, dtype=np.float32)
    for i, sid in enumerate(idx_list):
        cm = matched_control_mean(sid)
        if cm is not None:
            out[i] = cm
    return out

# 所有处理样本的 matched control 一次算好（train + val 场景）
treat_all = np.where(meta['role'].eq('treatment').values)[0]
print(f"[预计算] matched control for {len(treat_all)} 处理样本 ...", flush=True)
ctrl_all = get_ctrl_for(treat_all)
pos_of = {sid: i for i, sid in enumerate(treat_all)}
print(f"[预计算] 完成", flush=True)

# ---------- 特征张量 ----------
def make_x(idx):
    m = meta.iloc[idx]
    return {
        'bio': [torch.from_numpy(feats['strain_id'][idx]), torch.from_numpy(feats['chem_id'][idx]),
                torch.from_numpy(feats['chem_hash'][idx]), torch.from_numpy(feats['medium_onehot'][idx]),
                torch.from_numpy(feats['temp_norm'][idx]), torch.from_numpy(feats['time_feat'][idx]),
                torch.from_numpy(feats['sm_id'][idx]), torch.from_numpy(feats['ct_id'][idx])],
        'ctx': [torch.from_numpy(feats['src_id'][idx]), torch.from_numpy(feats['ins_id'][idx]),
                torch.from_numpy(feats['plt_id'][idx])],
        'seen': [torch.from_numpy(feats['chem_seen'][idx]), torch.from_numpy(feats['strain_seen'][idx])],
    }

# ---------- 训练集（处理样本；对照样本用于 B 分支监督，第一版合并 MSE）----------
train_treat = np.where(train_mask & meta['role'].eq('treatment').values)[0]
train_ctrl = np.where(train_mask & meta['role'].eq('control').values)[0]
print(f"[训练集] 处理 {len(train_treat)} / 对照 {len(train_ctrl)}", flush=True)

y_treat = torch.tensor(np.where(mask.astype(bool), y_log2, 0.0)[train_treat], device=DEV)  # 缺失填 0（mask 屏蔽）
m_treat = torch.tensor(mask[train_treat], device=DEV)
ctrl_treat = ctrl_all[[pos_of[sid] for sid in train_treat]]      # (n, P) NaN=无 matched
ctrl_treat_t = torch.tensor(ctrl_treat, device=DEV)

# 有 matched control 的样本用于 FC corr
fc_valid = np.isfinite(ctrl_treat).any(axis=1)
print(f"[FC] matched control 可用样本 {fc_valid.sum()}/{len(train_treat)}", flush=True)

# ---------- 模型 ----------
import importlib.util
_spec = importlib.util.spec_from_file_location("model04", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model.py")
_m04 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m04)
VCellModel = _m04.VCellModel
model = VCellModel(feats, P=P).to(DEV)
nparam = sum(p.numel() for p in model.parameters())
print(f"[模型] 参数量 {nparam/1e6:.2f}M", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
mse_loss = nn.MSELoss(reduction='none')

def fc_corr_loss(pred, yt, m, yc):
    """mask-aware Pearson corr loss on Δ (仅对 matched 样本)"""
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

def evaluate(epoch):
    model.eval()
    print(f"\n[Epoch {epoch} 验证] {'场景':<16}{'样本':>5}{'RMSE':>7}{'GlobalR2':>9}{'蛋白R2中位':>10}{'FC PCC':>8}", flush=True)
    for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
        idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
        if len(idx) == 0:
            continue
        yc = ctrl_all[[pos_of[sid] for sid in idx]]
        x = make_x(idx)
        with torch.no_grad():
            pred = model({k: [t.to(DEV) for t in v] for k, v in x.items()}).cpu().numpy()
        yt, m = y_log2[idx], mask[idx].astype(bool)
        valid = m & np.isfinite(pred) & np.isfinite(yc)
        ytv, ypv, mval = yt[valid], pred[valid], valid
        if mval.sum() == 0:
            continue
        rmse = float(np.sqrt(((yt[m] - pred[m]) ** 2).mean()))
        a, b = yt[mval], ypv
        g2 = 1 - ((a - b) ** 2).sum() / max(((a - a.mean()) ** 2).sum(), 1e-12)
        # 蛋白 R² 中位
        cnt = mval.sum(0); keep = cnt >= 3
        n = np.maximum(cnt.astype(float), 1)
        ytc = np.where(mval, yt, 0.0); ypc = np.where(mval, pred, 0.0)
        mt = ytc.sum(0) / n
        ss_tot = (((ytc - mt) ** 2) * mval).sum(0)
        ss_res = (((ytc - ypc) ** 2) * mval).sum(0)
        p2 = float(np.median(1 - ss_res / np.maximum(ss_tot, 1e-12)))
        # FC PCC（matched 样本）
        fc_ok = np.isfinite(yc) & m
        d_pred = (pred - yc)[fc_ok]; d_true = (yt - yc)[fc_ok]
        if len(d_pred) > 10:
            fc = float(np.corrcoef(d_pred, d_true)[0, 1])
        else:
            fc = float('nan')
        print(f"{scene:<20}{len(idx):>5}{rmse:>7.3f}{g2:>9.3f}{p2:>10.3f}{fc:>8.3f}", flush=True)

# ---------- 训练 ----------
EPOCHS = 150
BATCH = 256
best_p2 = -9
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = np.random.permutation(len(train_treat))
    total_loss = 0.0; n_batch = 0
    pbar = tqdm(range(0, len(perm), BATCH), desc=f"Ep {ep}/{EPOCHS}", leave=False, ncols=100)
    for i in pbar:
        b = perm[i:i + BATCH]
        x = make_x(train_treat[b])
        x = {k: [t.to(DEV) for t in v] for k, v in x.items()}
        pred = model(x)
        yt, m = y_treat[b], m_treat[b]
        loss_mse = (mse_loss(pred, yt) * m).sum() / m.sum()
        # FC corr（batch 内 matched 样本）
        yc = ctrl_treat_t[b]
        mask_fc = fc_valid[b]
        if mask_fc.any():
            loss_fc = fc_corr_loss(pred[mask_fc], yt[mask_fc], m[mask_fc], yc[mask_fc])
        else:
            loss_fc = torch.tensor(0.0, device=DEV)
        loss = loss_mse + 0.4 * loss_fc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item(); n_batch += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", mse=f"{loss_mse.item():.4f}")
    scheduler.step()
    if ep % 10 == 0 or ep == EPOCHS:
        evaluate(ep)
        # 简单保存（按 val_both 蛋白R2中位）
        torch.save(model.state_dict(), f"{DATA}/model_last.pt")
print("\n训练完成", flush=True)
