# -*- coding: utf-8 -*-
"""完整竞赛六模块评分（用val集模拟server评分）"""
import numpy as np, pandas as pd, pickle, torch, importlib.util

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float32)
mask = np.load(f'{DATA}/mask.npy').astype(np.float32)
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))
P = y_log2.shape[1]
ctx_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)
tr_y_nan = np.where(mask.astype(bool), y_log2, np.nan)
train_mask = meta['split_final'].eq('train').values

# ---------- matched control ----------
ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = (meta.iloc[ctrl_idx]['data_source'].astype(str) + '|' + meta.iloc[ctrl_idx]['instrument'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_idx]['Strains'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Medium'].astype(str) + '|' + meta.iloc[ctrl_idx]['Temperature'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['pert_time'].astype(str)).values
ctrl_lookup = {}
for k, pos in zip(ctrl_key, ctrl_idx):
    ctrl_lookup.setdefault(k, []).append(pos)

def matched_control_mean(sid):
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    if k not in ctrl_lookup: return np.full(P, np.nan)
    rows = ctrl_lookup[k]; cvals = tr_y_nan[rows]; cm = mask[rows] > 0
    with np.errstate(invalid='ignore'):
        return np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)

# ---------- 预计算训练集 μ_ctx, μ_drug（LOO版本，用于val集残差评估）----------
treat_all = np.where(meta['role'].eq('treatment').values)[0]
train_treat = np.where(train_mask & meta['role'].eq('treatment').values)[0]
pos_of = {sid: i for i, sid in enumerate(treat_all)}

# 计算所有处理样本的 Δ（用val的真实对照！模拟server）
ctrl_vals = np.full((len(treat_all), P), np.nan, dtype=np.float32)
for i, sid in enumerate(treat_all):
    cm = matched_control_mean(sid)
    if cm is not None: ctrl_vals[i] = cm
delta_all = tr_y_nan[treat_all] - ctrl_vals  # (N_treat, P) — Δ 真值

# 训练集 μ_ctx, μ_drug（LOO，仅用训练行）
chem_of = meta['perturbation_no_concentration'].values[treat_all]
ctx_key = (meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|'
           + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)).values[treat_all]
is_train_treat = np.isin(treat_all, train_treat)

mu_ctx_all = np.full((len(treat_all), P), np.nan)
mu_drug_all = np.full((len(treat_all), P), np.nan)

# 仅用训练集计算 μ（LOO）
for key, members in pd.Series(np.where(is_train_treat)[0], index=ctx_key[is_train_treat]).groupby(level=0):
    members = members.values
    if len(members) > 1:
        s = np.nansum(delta_all[members], axis=0)
        n = np.sum(~np.isnan(delta_all[members]), axis=0)
        for m in members:
            msk = ~np.isnan(delta_all[m])
            mu_ctx_all[m] = np.where(n > 0, (s - np.where(msk, delta_all[m], 0)) / np.maximum(n - msk.astype(float), 1), np.nan)

for key, members in pd.Series(np.where(is_train_treat)[0], index=chem_of[is_train_treat]).groupby(level=0):
    members = members.values
    if len(members) > 1:
        s = np.nansum(delta_all[members], axis=0)
        n = np.sum(~np.isnan(delta_all[members]), axis=0)
        for m in members:
            msk = ~np.isnan(delta_all[m])
            mu_drug_all[m] = np.where(n > 0, (s - np.where(msk, delta_all[m], 0)) / np.maximum(n - msk.astype(float), 1), np.nan)

# 对val集处理样本：用训练集全部数据的 μ（非LOO，等价于server）
tr_treat_idx = np.where(is_train_treat)[0]
mu_ctx_pool = {}; mu_drug_pool = {}
for key, members in pd.Series(tr_treat_idx, index=ctx_key[is_train_treat]).groupby(level=0):
    mu_ctx_pool[key] = np.nanmean(delta_all[members.values], axis=0)
for key, members in pd.Series(tr_treat_idx, index=chem_of[is_train_treat]).groupby(level=0):
    mu_drug_pool[key] = np.nanmean(delta_all[members.values], axis=0)

# ---------- Load models ----------
_s21 = importlib.util.spec_from_file_location('m21', r'D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v21.py')
_m21 = importlib.util.module_from_spec(_s21); _s21.loader.exec_module(_m21)
_s27 = importlib.util.spec_from_file_location('m27', r'D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v27.py')
_m27 = importlib.util.module_from_spec(_s27); _s27.loader.exec_module(_m27)

def load(path, cls, sa=False):
    m = cls(feats, P=P); m.load_state_dict(torch.load(f'{DATA}/{path}', map_location=DEV, weights_only=True))
    if sa: m.set_strain_avg()
    return m.to(DEV).eval()

m21 = load('model_v21.pt', _m21.VCellModel)
m21s43 = load('model_v21_s43.pt', _m21.VCellModel)
m21s44 = load('model_v21_s44.pt', _m21.VCellModel)
m27 = load('model_v27_best.pt', _m27.VCellModel, sa=True)
models = [(m21,'v21'),(m21s43,'v21'),(m21s44,'v21'),(m27,'v27')]

def ensemble_pred(idx):
    preds = []
    for m, tag in models:
        x = {'bio': [torch.from_numpy(feats['strain_id'][idx]), torch.from_numpy(feats['chem_id'][idx]),
                     torch.from_numpy(feats['chem_hash'][idx]), torch.from_numpy(feats['medium_onehot'][idx]),
                     torch.from_numpy(feats['temp_norm'][idx]), torch.from_numpy(feats['time_feat'][idx]),
                     torch.from_numpy(feats['sm_id'][idx]), torch.from_numpy(feats['ct_id'][idx])],
             'ctx': [torch.from_numpy(feats['src_id'][idx]), torch.from_numpy(feats['ins_id'][idx]), torch.from_numpy(feats['plt_id'][idx])],
             'seen': [torch.from_numpy(feats['chem_seen'][idx]), torch.from_numpy(feats['strain_seen'][idx])]}
        if tag == 'v27': x['ctx_prior'] = torch.from_numpy(ctx_all[idx])
        with torch.no_grad():
            xg = {k: (v.to(DEV) if k == 'ctx_prior' else [t.to(DEV) for t in v]) for k, v in x.items()}
            preds.append(m(xg).cpu().numpy())
    return np.mean(preds, axis=0)

# ---------- 辅助函数 ----------
def safe_pcc(a, b):
    """PCC, 忽略 NaN/inf"""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10: return 0.0
    a, b = a[ok], b[ok]
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0

def safe_r2(y_pred, y_true, ok_mask):
    """R² on valid positions"""
    if ok_mask.sum() < 3: return 0.0
    a, b = y_pred[ok_mask], y_true[ok_mask]
    ss_res = ((a - b) ** 2).sum()
    ss_tot = ((b - b.mean()) ** 2).sum()
    if ss_tot < 1e-12: return 0.0
    r2 = 1 - ss_res / ss_tot
    return float(r2) if np.isfinite(r2) else 0.0

# ---------- 完整六模块评分 ----------
print("=" * 100)
print("竞赛六模块完整评分（val集模拟，新集成 3×v2.1+v2.7）")
print("=" * 100)

all_scenes = ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']
scene_weight = {'val_chem_only': 1.0, 'val_strain_only': 1.0, 'val_both': 0.5, 'val_time': 0.5}

# 预计算所有预测
preds_cache = {}
for scene in all_scenes:
    idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    if len(idx) > 0:
        preds_cache[scene] = (idx, ensemble_pred(idx))

scores = {}
for scene in all_scenes:
    if scene not in preds_cache: continue
    idx, yp = preds_cache[scene]
    yt = y_log2[idx]; mt = mask[idx].astype(bool)
    
    # 获取该场景在 treat_all 中的位置
    scene_pos = np.array([pos_of[s] for s in idx])
    yc = ctrl_vals[scene_pos]  # 真实对照值
    dt = delta_all[scene_pos]  # 真实 Δ
    dp = yp - yc               # 预测 Δ
    
    # ---- 模块1: 匹配对照原始FC (25%) ----
    # PCC(Δ_pred, Δ_true) — 全蛋白向量展开
    fc_ok = np.isfinite(dp) & np.isfinite(dt) & mt
    mod1 = safe_pcc(dp[fc_ok].ravel(), dt[fc_ok].ravel())
    
    # ---- 模块2: 绝对保真度 (20%) ----
    # 逐蛋白R²中位（聚合蛋白轴）+ 逐样本R²中位
    # 逐蛋白
    p_r2s = []
    for p in range(P):
        ok = mt[:, p] & np.isfinite(yp[:, p]) & np.isfinite(yt[:, p])
        if ok.sum() >= 3:
            p_r2s.append(safe_r2(yp[:, p], yt[:, p], ok))
    prot_r2 = float(np.median(p_r2s)) if p_r2s else 0.0
    
    # 逐样本
    s_r2s = []
    for s in range(len(idx)):
        ok = mt[s] & np.isfinite(yp[s]) & np.isfinite(yt[s])
        if ok.sum() >= 10:
            s_r2s.append(safe_r2(yp[s], yt[s], ok))
    samp_r2 = float(np.median(s_r2s)) if s_r2s else 0.0
    
    mod2 = 0.5 * max(prot_r2, 0) + 0.5 * max(samp_r2, 0)  # R²可为负，截断
    
    # ---- 模块3: 上下文均值残差 (20%) ----
    # PCC(Δ_pred − μ_ctx, Δ_true − μ_ctx)
    mu_c = np.array([mu_ctx_pool.get(ctx_key[p], np.zeros(P)) for p in scene_pos])
    mod3 = safe_pcc((dp - mu_c)[fc_ok].ravel(), (dt - mu_c)[fc_ok].ravel())
    
    # ---- 模块4: 药物均值残差 (20%) ----
    mu_d = np.array([mu_drug_pool.get(chem_of[p], np.zeros(P)) for p in scene_pos])
    mod4 = safe_pcc((dp - mu_d)[fc_ok].ravel(), (dt - mu_d)[fc_ok].ravel())
    
    # ---- 模块5: 双重未知/时间外推 (10%) ----
    if scene in ('val_both', 'val_time'):
        # test_both: FC + 绝对保真度; test_time: 绝对保真度 + FC
        if scene == 'val_both':
            mod5 = 0.5 * mod1 + 0.5 * prot_r2
        else:
            mod5 = 0.7 * prot_r2 + 0.3 * mod1
        mod5 = max(mod5, 0)
    else:
        mod5 = None  # 不适用
    
    # ---- 模块6: 高效应蛋白/DEP (5%) ----
    # |Δ_true| > 1 → 方向准确率 + 高效应PCC
    hi = np.abs(dt) > 1
    dep_acc = 0.0; dep_pcc = 0.0
    if hi.sum() > 0:
        dp_hi = dp[hi & fc_ok]; dt_hi = dt[hi & fc_ok]
        if len(dp_hi) > 10:
            dep_acc = float((np.sign(dp_hi) == np.sign(dt_hi)).mean())
            dep_pcc = safe_pcc(dp_hi.ravel(), dt_hi.ravel())
    mod6 = 0.5 * dep_acc + 0.5 * max(dep_pcc, 0)
    
    # 汇总
    total = 0.25*mod1 + 0.20*mod2 + 0.20*mod3 + 0.20*mod4
    if mod5 is not None:
        total += 0.10*mod5
    total += 0.05*mod6
    
    scores[scene] = (mod1, mod2, mod3, mod4, mod5, mod6, total)

# 打印
hdr = f"{'场景':<16}{'M1:FC(25%)':>10}{'M2:绝对(20%)':>12}{'M3:ctx残差(20%)':>15}{'M4:drug残差(20%)':>15}{'M5:双盲(10%)':>12}{'M6:DEP(5%)':>11}{'总分':>8}"
print(hdr)
print("-" * 115)
for scene in all_scenes:
    if scene not in scores: continue
    m1,m2,m3,m4,m5,m6,t = scores[scene]
    m5s = f"{m5:.3f}" if m5 is not None else "N/A"
    print(f"{scene:<16}{m1:>10.3f}{m2:>12.3f}{m3:>15.3f}{m4:>15.3f}{m5s:>12}{m6:>11.3f}{t:>8.3f}")

# 加权总分
print()
print("各场景加权总分 (val_chem×1 + val_strain×1 + val_both×0.5 + val_time×0.5):")
total_weighted = 0; w_sum = 0
for scene, w in [('val_chem_only',1),('val_strain_only',1),('val_both',0.5),('val_time',0.5)]:
    if scene in scores:
        total_weighted += scores[scene][-1] * w
        w_sum += w
print(f"  加权总分 = {total_weighted/w_sum:.4f}")

print()
print("=" * 100)
print("关键提醒:")
print("  M1(FC PCC) = 服务器用真实y_control计算PCC(Δ_pred, Δ_true)")
print("  M3(上下文残差) = 扣除同条件下训练药物的平均响应")
print("  M4(药物残差) = 扣除同药物在不同上下文的平均响应")
print("  65%的分数(M1+M3+M4)围绕Δ=处理−对照展开")
print("  本次模拟用val集的真实对照值，与server口径一致")
print("=" * 100)
