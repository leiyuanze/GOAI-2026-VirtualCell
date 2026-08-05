# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 05p 训练 v2.8（Strain + Compound Dropout）
基于 v2.7 模型 + 训练时随机遮蔽菌株/化合物信息，模拟 unseen 场景
论文依据：TxPert (Nature Biotech 2025) leave-one-out策略
  - 30% 概率遮蔽菌株信息（strain_id→-1, strain_seen→0, ctx_prior→gmean）
  - 20% 概率遮蔽化合物信息（chem_id→-1, chem_seen→0, chem_hash→0）
  - 2% 概率同时遮蔽（模拟 val_both）
模型必须学会在信息缺失时利用上下文（medium/temp/time）做出合理预测
"""
import os, sys, pickle, numpy as np, pandas as pd
import torch, torch.nn as nn
from tqdm import tqdm

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
torch.manual_seed(_SEED); np.random.seed(_SEED)
print(f"[设备] {DEV} | seed={_SEED} | 策略=退火Strain+Compound Dropout(0→20%/10%)", flush=True)

meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy").astype(np.float32)
mask = np.load(f"{DATA}/mask.npy").astype(np.float32)
with open(f"{DATA}/feats.pkl", 'rb') as f:
    feats = pickle.load(f)
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values

# ctx_prior base（NaN→0）
ctx_prior_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)
gmean = feats['gmean'].astype(np.float32)

# ---------- matched control ----------
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
print(f"[训练集] 处理 {len(train_treat)} + 对照 {len(train_ctrl)} = {len(train_idx)}", flush=True)

y_train = np.where(mask.astype(bool), y_log2, 0.0)[train_idx].astype(np.float32)
m_train = mask[train_idx].astype(np.float32)

# 存储原始特征（用于 dropout 恢复）
_strain_id_base = feats['strain_id'][train_idx].copy()
_chem_id_base = feats['chem_id'][train_idx].copy()
_chem_hash_base = feats['chem_hash'][train_idx].copy()
_strain_seen_base = feats['strain_seen'][train_idx].copy()
_chem_seen_base = feats['chem_seen'][train_idx].copy()
_ctx_prior_base = ctx_prior_all[train_idx].copy()

# ---------- 组件监督目标（LOO）----------
delta_treat = np.full((len(treat_all), P), np.nan)
for i in range(len(treat_all)):
    delta_treat[i] = tr_y_nan[treat_all[i]] - ctrl_all[i]
chem_of = meta['perturbation_no_concentration'].values[treat_all]
ctx_of = (meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|'
          + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)).values[treat_all]
print("[组件监督] 计算 LOO μ_ctx / μ_drug ...", flush=True)
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
print("[组件监督] 完成", flush=True)

treat_pos = np.array([pos_of[s] for s in train_treat])
resid_ctx = delta_treat[treat_pos] - mu_ctx[treat_pos]
resid_drug = delta_treat[treat_pos] - mu_drug[treat_pos]
resid_ctx = np.where(np.isnan(resid_ctx), 0.0, resid_ctx).astype(np.float32)
resid_drug = np.where(np.isnan(resid_drug), 0.0, resid_drug).astype(np.float32)
mask_resid = np.isfinite(delta_treat[treat_pos]) & np.isfinite(mu_ctx[treat_pos]) & (mu_ctx[treat_pos] != 0)
mask_resid_drug = np.isfinite(delta_treat[treat_pos]) & np.isfinite(mu_drug[treat_pos]) & (mu_drug[treat_pos] != 0)

# ---------- 特征构造（含 dropout）----------
def apply_dropout(idx_in_batch, drop_strain_prob=0.30, drop_chem_prob=0.20, drop_both_prob=0.02):
    """对 batch 内样本施加随机遮蔽。返回修改后的特征副本"""
    n = len(idx_in_batch)
    sid = _strain_id_base[idx_in_batch].copy()
    cid = _chem_id_base[idx_in_batch].copy()
    chash = _chem_hash_base[idx_in_batch].copy()
    sseen = _strain_seen_base[idx_in_batch].copy()
    cseen = _chem_seen_base[idx_in_batch].copy()
    cprior = _ctx_prior_base[idx_in_batch].copy()

    rand = np.random.random(n)
    drop_s = rand < drop_strain_prob
    drop_c = (rand >= drop_strain_prob) & (rand < drop_strain_prob + drop_chem_prob)
    drop_both = (rand >= drop_strain_prob + drop_chem_prob) & (rand < drop_strain_prob + drop_chem_prob + drop_both_prob)

    # strain dropout
    sid[drop_s | drop_both] = -1
    sseen[drop_s | drop_both] = 0.0
    cprior[drop_s | drop_both] = gmean  # 遮蔽菌株 → ctx_prior 用全局均值

    # compound dropout
    cid[drop_c | drop_both] = -1
    cseen[drop_c | drop_both] = 0.0
    chash[drop_c | drop_both] = 0.0

    return sid, cid, chash, sseen, cseen, cprior

def make_x_dropout(pos_idx, drop_strain_prob=0.30, drop_chem_prob=0.20, drop_both_prob=0.02):
    """pos_idx: 在 train_idx 中的位置。feats 用 train_idx[pos_idx] 索引全量数据"""
    full_idx = train_idx[pos_idx]  # 映射到全量数据索引
    sid, cid, chash, sseen, cseen, cprior = apply_dropout(pos_idx, drop_strain_prob, drop_chem_prob, drop_both_prob)
    return {
        'bio': [torch.from_numpy(sid), torch.from_numpy(cid),
                torch.from_numpy(chash), torch.from_numpy(feats['medium_onehot'][full_idx]),
                torch.from_numpy(feats['temp_norm'][full_idx]), torch.from_numpy(feats['time_feat'][full_idx]),
                torch.from_numpy(feats['sm_id'][full_idx]), torch.from_numpy(feats['ct_id'][full_idx])],
        'ctx': [torch.from_numpy(feats['src_id'][full_idx]), torch.from_numpy(feats['ins_id'][full_idx]),
                torch.from_numpy(feats['plt_id'][full_idx])],
        'seen': [torch.from_numpy(cseen), torch.from_numpy(sseen)],
        'ctx_prior': torch.from_numpy(cprior),
    }

def make_x_eval(idx):
    """构造特征，不做 dropout"""
    return {
        'bio': [torch.from_numpy(feats['strain_id'][idx]), torch.from_numpy(feats['chem_id'][idx]),
                torch.from_numpy(feats['chem_hash'][idx]), torch.from_numpy(feats['medium_onehot'][idx]),
                torch.from_numpy(feats['temp_norm'][idx]), torch.from_numpy(feats['time_feat'][idx]),
                torch.from_numpy(feats['sm_id'][idx]), torch.from_numpy(feats['ct_id'][idx])],
        'ctx': [torch.from_numpy(feats['src_id'][idx]), torch.from_numpy(feats['ins_id'][idx]),
                torch.from_numpy(feats['plt_id'][idx])],
        'seen': [torch.from_numpy(feats['chem_seen'][idx]), torch.from_numpy(feats['strain_seen'][idx])],
        'ctx_prior': torch.from_numpy(ctx_prior_all[idx]),
    }

# ---------- 模型 ----------
import importlib.util
_spec = importlib.util.spec_from_file_location("m27", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v27.py")
_m27 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m27)
model = _m27.VCellModel(feats, P=P).to(DEV)
model.set_strain_avg()
print(f"[模型] 参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M (v2.7 + Dropout训练)", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, min_lr=5e-6)
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
    print(f"\n[Epoch {ep}] {'场景':<20}{'样本':>5}{'RMSE':>7}{'GlobalR2':>9}{'蛋白R2中位':>10}{'FC PCC':>8}", flush=True)
    for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
        idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
        if len(idx) == 0:
            continue
        x = make_x_eval(idx)
        with torch.no_grad():
            xd = {}
            for k, v in x.items():
                if k == 'ctx_prior':
                    xd[k] = v.to(DEV)
                else:
                    xd[k] = [t.to(DEV) for t in v]
            pred = model(xd).cpu().numpy()
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

EPOCHS, BATCH = 200, 256
best_score = float('inf')
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = np.random.permutation(len(train_idx))
    total_loss = 0.0; n_batch = 0
    # 退火 dropout：前 80 epoch 0%，之后线性增长到 20% strain / 10% chem
    progress = min(1.0, max(0.0, (ep - 80) / 80.0))
    drop_s = 0.20 * progress
    drop_c = 0.10 * progress
    drop_b = 0.02 * progress
    pbar = tqdm(range(0, len(perm), BATCH), desc=f"Ep {ep}/{EPOCHS}", leave=False, ncols=100)
    for i in pbar:
        b = perm[i:i + BATCH]
        x = make_x_dropout(b, drop_s, drop_c, drop_b)
        x_gpu = {k: (v.to(DEV) if k == 'ctx_prior' else [t.to(DEV) for t in v]) for k, v in x.items()}
        pred = model(x_gpu)
        yt, m = y_train_t[b], m_train_t[b]
        loss_mse = (mse_loss(pred, yt) * m).sum() / m.sum()
        loss = loss_mse
        # FC corr（仅处理样本）
        is_treat = treat_in_train[b]
        if is_treat.any():
            pos = np.where(is_treat)[0]
            yc = y_ctrl_all_t[b[pos]]
            fc_ok = torch.isfinite(yc).any(dim=1)
            if fc_ok.any():
                loss += 0.4 * fc_corr_loss(pred[pos][fc_ok], yt[pos][fc_ok], m[pos][fc_ok], yc[fc_ok])
        # 组件级监督
        if is_treat.any():
            pos = np.where(is_treat)[0]
            yB, yS, yC, yT = model.components({k: (v[pos].to(DEV) if k == 'ctx_prior' else [t[pos].to(DEV) for t in v])
                                                for k, v in x_gpu.items()})
            loss += 0.15 * corr_loss(yC, resid_ctx_t[b[pos]], mask_resid_t[b[pos]])
            loss += 0.15 * corr_loss(yT, resid_drug_t[b[pos]], mask_resid_drug_t[b[pos]])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item(); n_batch += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", mse=f"{loss_mse.item():.4f}")
    avg_loss = total_loss / max(n_batch, 1)
    scheduler.step(avg_loss)
    if ep % 10 == 0 or ep == EPOCHS:
        evaluate(ep)
        torch.save(model.state_dict(), f"{DATA}/model_v28.pt")
    if avg_loss < best_score:
        best_score = avg_loss
        torch.save(model.state_dict(), f"{DATA}/model_v28_best.pt")
    lr_now = optimizer.param_groups[0]['lr']
    print(f"  avg_loss={avg_loss:.4f}  lr={lr_now:.2e}  best={best_score:.4f}  drop_s={drop_s:.1%} drop_c={drop_c:.1%}", flush=True)

print("\n训练完成（v2.8 Strain+Compound Dropout）", flush=True)
