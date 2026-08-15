# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 05 训练 v4.6（评测权重再平衡 + pairwise 排序 loss + pool μ 残差监督）
针对前 4 个失败实验的根因重构：
1) 评测权重倒挂：M2(绝对) 只占 20%，但训练 MSE 权重 0.5 主导；
   Δ 形状指标(M1+M3+M4) 占 65% 却只有 0.25。→ MSE 降到 0.3，FC 升到 0.35
2) v4.3 排序 loss 实现错误：全局 corr(|Δ|) 与 fc_corr 冗余。
   → 改为 pairwise Learning-to-Rank：采样蛋白对，|Δ_true| 大的预测 |Δ_pred| 也应大
     （直接对齐 M6 DEP：高效应蛋白排序）
3) v4.4 残差监督用 LOO μ（噪声大、口径不一致）。→ 改用训练集 pool μ（与评测一致）
4) v4.2 PPI 约束 (Δ_i−Δ_j)² 强制相等是错误先验。
   → 改为 PPI 协同正则：约束互作蛋白的 Δ 相关（点积），允许量级差异
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

train_treat = np.where(train_mask & meta['role'].eq('treatment').values)[0]
train_ctrl = np.where(train_mask & meta['role'].eq('control').values)[0]
train_idx = np.concatenate([train_treat, train_ctrl])
print(f"[训练集] 处理 {len(train_treat)} + 对照 {len(train_ctrl)} = {len(train_idx)}", flush=True)

y_train = np.where(mask.astype(bool), y_log2, 0.0)[train_idx].astype(np.float32)
m_train = mask[train_idx].astype(np.float32)
ctx_prior_train = ctx_prior_all[train_idx]

# ---------- 组件监督目标：pool μ（与评测口径一致，非 LOO）----------
# 评测 M3/M4 的 μ_ctx/μ_drug 是"训练集真值常数"（pool，非 LOO）
delta_treat = np.full((len(treat_all), P), np.nan)
for i in range(len(treat_all)):
    delta_treat[i] = tr_y_nan[treat_all[i]] - ctrl_all[i]
chem_of = meta['perturbation_no_concentration'].values[treat_all]
ctx_of = (meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|'
          + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)).values[treat_all]
print("[组件监督] 计算 pool μ_ctx / μ_drug（与评测一致）...", flush=True)
is_train_treat = np.isin(treat_all, train_treat)
tr_treat_idx = np.where(is_train_treat)[0]
mu_ctx_pool = {}; mu_drug_pool = {}
for key, members in pd.Series(tr_treat_idx, index=ctx_of[is_train_treat]).groupby(level=0):
    mu_ctx_pool[key] = np.nanmean(delta_treat[members.values], axis=0)
for key, members in pd.Series(tr_treat_idx, index=chem_of[is_train_treat]).groupby(level=0):
    mu_drug_pool[key] = np.nanmean(delta_treat[members.values], axis=0)
# 映射到 treat_all 行
mu_ctx = np.full((len(treat_all), P), np.nan)
mu_drug = np.full((len(treat_all), P), np.nan)
for t in tr_treat_idx:
    rel = t  # tr_treat_idx 本身就是 treat_all 中的相对位置
    mu_ctx[rel] = mu_ctx_pool.get(ctx_of[rel], np.full(P, np.nan))
    mu_drug[rel] = mu_drug_pool.get(chem_of[rel], np.full(P, np.nan))
print("[组件监督] 完成", flush=True)

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
print(f"[GPU] 特征预加载完成，显存 {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

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

# ---------- 模型 ----------
import importlib.util
_spec = importlib.util.spec_from_file_location("m37", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v37.py")
_m37 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m37)
model = _m37.VCellModel(feats, P=P).to(DEV)
model.set_strain_avg()
print(f"[模型] 参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M (v4.6 评测权重再平衡+pairwise排序+pool μ)", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)
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

# ★ pairwise 排序 loss（v4.6 核心，对齐 M6 DEP 高效应蛋白检出）
# 采样蛋白对 (i,j)：|Δ_true_i| > |Δ_true_j| 的样本对，要求 |Δ_pred_i| > |Δ_pred_j|
# 用 hinge 形式：loss = relu(margin - (|Δp_i| - |Δp_j|) * sign(真值差))
def rank_pair_loss(pred, yt, m, yc, n_pairs=512, margin=0.05):
    yc = yc.to(pred.device)
    B = pred.shape[0]
    v = m.bool() & torch.isfinite(yc)
    if v.sum() < 10:
        return torch.tensor(0.0, device=pred.device)
    dp = torch.where(v, pred - yc, torch.zeros_like(pred))
    dt = torch.where(v, yt - yc, torch.zeros_like(pred))
    ap = dp.abs(); at = dt.abs()
    # 采样有效蛋白对
    idx = torch.nonzero(v)
    if len(idx) < 2:
        return torch.tensor(0.0, device=pred.device)
    perm = torch.randperm(len(idx), device=pred.device)[:n_pairs * 2]
    sel = idx[perm]  # (n, 2): (sample, protein)
    a = sel[:n_pairs]; b = sel[n_pairs:2 * n_pairs]
    # 保证 a 的 |Δ_true| > b 的 |Δ_true|
    at_a = at[a[:, 0], a[:, 1]]
    at_b = at[b[:, 0], b[:, 1]]
    ap_a = ap[a[:, 0], a[:, 1]]
    ap_b = ap[b[:, 0], b[:, 1]]
    diff_true = at_a - at_b
    keep = diff_true.abs() > 1e-4  # 只保留真值有差异的对
    if keep.sum() < 10:
        return torch.tensor(0.0, device=pred.device)
    diff_true = diff_true[keep]
    ap_a = ap_a[keep]; ap_b = ap_b[keep]
    sign = torch.sign(diff_true)
    diff_pred = (ap_a - ap_b) * sign
    loss = torch.relu(margin - diff_pred).mean()
    return loss

# ★ PPI 协同正则（修正 v4.2）：互作蛋白 Δ 应协同（点积正），非强制相等
def ppi_coop_loss(pred, m, yc, ppi_i, ppi_j):
    yc = yc.to(pred.device)
    v = m.bool() & torch.isfinite(yc)
    dp = torch.where(v, pred - yc, torch.zeros_like(pred))
    d_i = dp[:, ppi_i]
    d_j = dp[:, ppi_j]
    m_i = v[:, ppi_i].float(); m_j = v[:, ppi_j].float()
    mask_e = (m_i * m_j)  # 边两端都有观测
    num = (d_i * d_j * mask_e).sum()
    den = mask_e.sum().clamp(min=1)
    # 期望正相关（协同），负的点积是惩罚
    return -num / den

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
mu_ctx_t = torch.tensor(mu_ctx[treat_pos], device=DEV)
mu_drug_t = torch.tensor(mu_drug[treat_pos], device=DEV)
resid_ctx_t = torch.tensor(resid_ctx, device=DEV)
resid_drug_t = torch.tensor(resid_drug, device=DEV)
mask_resid_t = torch.tensor(mask_resid, device=DEV)
mask_resid_drug_t = torch.tensor(mask_resid_drug, device=DEV)
treat_in_train = np.isin(train_idx, train_treat)

# PPI 边（top 5000）
ppi_edges = feats.get('ppi_edges_top', np.zeros((0, 2), dtype=np.int64))
if len(ppi_edges) > 5000:
    rng = np.random.RandomState(42)
    ppi_edges = ppi_edges[rng.choice(len(ppi_edges), 5000, replace=False)]
ppi_i = torch.tensor(ppi_edges[:, 0], device=DEV, dtype=torch.long)
ppi_j = torch.tensor(ppi_edges[:, 1], device=DEV, dtype=torch.long)
print(f"[预计算] PPI 边 {len(ppi_edges)} 条 (top5000)", flush=True)

from sklearn.decomposition import TruncatedSVD
print("[预计算] 蛋白共表达主成分 ...", flush=True)
X_tr_full = np.where(mask[train_mask].astype(bool), y_log2[train_mask], np.nan)
gmean_tr = np.nanmean(X_tr_full, axis=0)
Xc_graph = np.where(mask[train_mask].astype(bool), y_log2[train_mask] - gmean_tr, 0.0)
svd = TruncatedSVD(n_components=64, random_state=42)
svd.fit(Xc_graph)
V_graph = svd.components_.T.astype(np.float32)
V_graph = V_graph / (np.linalg.norm(V_graph, axis=0, keepdims=True) + 1e-8)
V_graph_t = torch.tensor(V_graph, device=DEV)

EPOCHS, BATCH = 120, 128
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
        # ★ 评测权重再平衡：MSE 0.3（M2 只占 20%），FC 0.35（M1 25%+形状）
        loss = 0.3 * loss_mse
        loss_fc_val = torch.tensor(0.0, device=DEV)
        loss_rank_val = torch.tensor(0.0, device=DEV)
        loss_ppi_val = torch.tensor(0.0, device=DEV)
        loss_resid_ctx_val = torch.tensor(0.0, device=DEV)
        loss_resid_drug_val = torch.tensor(0.0, device=DEV)
        is_treat = treat_in_train[b]
        if is_treat.any():
            pos = np.where(is_treat)[0]
            yc = y_ctrl_all_t[b[pos]]
            fc_ok = torch.isfinite(yc).any(dim=1)
            if fc_ok.any():
                loss_fc_val = fc_corr_loss(pred[pos][fc_ok], yt[pos][fc_ok], m[pos][fc_ok], yc[fc_ok])
                loss += 0.35 * loss_fc_val
                # pairwise 排序 loss（0.15）
                loss_rank_val = rank_pair_loss(pred[pos][fc_ok], yt[pos][fc_ok], m[pos][fc_ok], yc[fc_ok])
                loss += 0.15 * loss_rank_val
                # PPI 协同正则（0.05）
                loss_ppi_val = ppi_coop_loss(pred[pos][fc_ok], m[pos][fc_ok], yc[fc_ok], ppi_i, ppi_j)
                loss += 0.05 * loss_ppi_val
                # ★ pool μ 残差形状监督（0.1，评测 M3/M4 口径一致）
                loss_resid_ctx_val = corr_loss(pred[pos][fc_ok], resid_ctx_t[b[pos]][fc_ok],
                                               mask_resid_t[b[pos]][fc_ok])
                loss_resid_drug_val = corr_loss(pred[pos][fc_ok], resid_drug_t[b[pos]][fc_ok],
                                                mask_resid_drug_t[b[pos]][fc_ok])
                loss += 0.1 * loss_resid_ctx_val + 0.1 * loss_resid_drug_val
        # 组件级监督（保持）
        if is_treat.any():
            pos = np.where(is_treat)[0]
            yB, yS, yC, yT = model.components({k: (v[pos] if k in ('ctx_prior', 'chem_morgan') else [t[pos] for t in v])
                                                for k, v in x.items()})
            loss_ctx_val = corr_loss(yC, resid_ctx_t[b[pos]], mask_resid_t[b[pos]])
            loss_drug_val = corr_loss(yT, resid_drug_t[b[pos]], mask_resid_drug_t[b[pos]])
            loss += 0.2 * loss_ctx_val + 0.2 * loss_drug_val
        loss_prot = per_protein_corr_loss(pred, yt, m)
        loss_graph = graph_reg_loss(pred, m, V_graph_t)
        loss += 0.1 * loss_prot + 0.05 * loss_graph
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item(); n_batch += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", mse=f"{loss_mse.item():.4f}", fc=f"{loss_fc_val.item():.3f}", rank=f"{loss_rank_val.item():.4f}")
    avg_loss = total_loss / max(n_batch, 1)
    scheduler.step(avg_loss)
    lr_now = optimizer.param_groups[0]['lr']
    if ep % 10 == 0 or ep == EPOCHS:
        evaluate(ep)
        torch.save(model.state_dict(), f"{DATA}/model_v46_{_SEED}.pt")
    if avg_loss < best_score:
        best_score = avg_loss
        torch.save(model.state_dict(), f"{DATA}/model_v46_{_SEED}_best.pt")
    print(f"  avg_loss={avg_loss:.4f}  lr={lr_now:.2e}", flush=True)

print("\n训练完成", flush=True)
