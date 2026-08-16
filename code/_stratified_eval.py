# -*- coding: utf-8 -*-
"""
gpt1 评估补强（§七）：bootstrap CI + 按蛋白/化合物/菌株分层评估
在 val 四场景上评估 0.75×v37 + 0.25×v5.2 集成（seed 42 复现），输出：
  1) 每场景 FC PCC / 蛋白 R² 的 mean ± 95% bootstrap CI（样本重采样 1000 次）
  2) 按蛋白类别分层（高/低丰度、高/低变异、GO 覆盖/无 GO）的蛋白 R²
  3) 按化合物相似度分层（val_chem_only 高/中/低）的 FC PCC
  4) 按菌株遗传距离分层（val_strain_only/val_both 近/远）的 FC PCC
全部统计量 train-only。
"""
import pickle, numpy as np, pandas as pd, torch, os
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
train_mask = meta['split_final'].eq('train').values

# ---------- 对照（全量观测对照，评估口径） ----------
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

# ---------- 模型 ----------
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
m50 = _m50.VCellModel(feats, P=P, response_basis=U_basis).to(DEV)
m50.load_state_dict(torch.load(f"{DATA}/model_v50_42_best.pt", map_location=DEV, weights_only=True))
m50.set_strain_avg(); m50.eval()

SCENES = ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']

# ---------- 蛋白分层元数据（train-only 统计量） ----------
tr_y = tr_y_nan[train_mask]
prot_mean = np.nanmean(tr_y, axis=0)
prot_std = np.nanstd(tr_y, axis=0)
q33, q66 = np.nanquantile(prot_mean, [1/3, 2/3]), np.nanquantile(prot_std, [1/3, 2/3])
ab_hi = prot_mean > q33[1]; ab_lo = prot_mean < q33[0]
var_hi = prot_std > q66[1]; var_lo = prot_std < q66[0]
go_prots = set()
if os.path.exists(f"{DATA}/uniprot_go.tsv"):
    gd = pd.read_csv(f"{DATA}/uniprot_go.tsv", sep='\t', header=None)
    go_prots = set(gd[0].astype(str))
prot_names = [l.strip() for l in open(f"{DATA}/prot_names.txt")]
prot_has_go = np.array([p in go_prots for p in prot_names])

# ---------- 预测 ----------
pred_all = {}
with torch.no_grad():
    for scene in SCENES:
        idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
        x = make_x(idx)
        p50 = m50(x).cpu().numpy()
        p37 = m37(x).cpu().numpy()
        pred_all[scene] = 0.75 * p37 + 0.25 * p50
        print(f"预测 {scene}: n={len(idx)}", flush=True)

# ---------- 指标 ----------
def fc_pcc(pred, yt, yc, m):
    ok = np.isfinite(yc) & m.astype(bool) & np.isfinite(pred)
    d_pred, d_true = (pred - yc)[ok], (yt - yc)[ok]
    if len(d_pred) < 10 or d_pred.std() < 1e-12 or d_true.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(d_pred, d_true)[0, 1])

def prot_r2(pred, yt, m, prot_sel=None):
    r2s = []
    for pp in range(yt.shape[1]):
        if prot_sel is not None and not prot_sel[pp]:
            continue
        ok = m[:, pp].astype(bool) & np.isfinite(pred[:, pp])
        if ok.sum() >= 3:
            a, b = pred[:, pp][ok], yt[:, pp][ok]
            ss_res = ((a - b) ** 2).sum(); ss_tot = ((b - b.mean()) ** 2).sum()
            if ss_tot > 1e-12:
                r2s.append(1 - ss_res / ss_tot)
    return float(np.median(r2s)) if r2s else float('nan')

def bootstrap_ci(values, n=1000, alpha=0.05):
    """values: (n_obs, ) 每观测一个标量指标（如逐样本 FC 方向 or 蛋白 R² 需要重采样维度）"""
    rng = np.random.default_rng(42)
    vals = values[~np.isnan(values)]
    if len(vals) < 10:
        return float('nan'), float('nan'), float('nan')
    # 蛋白 R² 分层下用逐蛋白 bootstrap
    boot = np.array([np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(n)])
    return float(np.median(boot)), float(np.quantile(boot, alpha/2)), float(np.quantile(boot, 1-alpha/2))

def prot_r2_array(pred, yt, m):
    r2s = []
    for pp in range(yt.shape[1]):
        ok = m[:, pp].astype(bool) & np.isfinite(pred[:, pp])
        if ok.sum() >= 3:
            a, b = pred[:, pp][ok], yt[:, pp][ok]
            ss_res = ((a - b) ** 2).sum(); ss_tot = ((b - b.mean()) ** 2).sum()
            if ss_tot > 1e-12:
                r2s.append(1 - ss_res / ss_tot)
    return np.array(r2s)

print("\n=== 一、val 四场景主指标 + bootstrap 95% CI（样本重采样 1000 次）===")
for scene in SCENES:
    idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    pred, yt, m = pred_all[scene], y_log2[idx], mask[idx]
    yc = ctrl_all[[pos_of[s] for s in idx]]
    fc = fc_pcc(pred, yt, yc, m)
    # FC：按样本块重采样（保持 2D，避免展平后索引错误）
    fc_ok_mat = np.isfinite(yc) & m.astype(bool) & np.isfinite(pred)
    dp2, dt2 = pred - yc, yt - yc
    rng = np.random.default_rng(7)
    boot_fc = []
    for _ in range(1000):
        ids = rng.choice(len(idx), len(idx), replace=True)
        dps = np.concatenate([dp2[i][fc_ok_mat[i]] for i in ids])
        dts = np.concatenate([dt2[i][fc_ok_mat[i]] for i in ids])
        if len(dps) > 10 and dps.std() > 1e-12 and dts.std() > 1e-12:
            boot_fc.append(np.corrcoef(dps, dts)[0, 1])
    boot_fc = np.array(boot_fc)
    r2_arr = prot_r2_array(pred, yt, m)
    bmed, blo, bhi = bootstrap_ci(r2_arr)
    print(f"{scene:<18} FC PCC={fc:.3f} (95%CI [{np.quantile(boot_fc,0.025):.3f},{np.quantile(boot_fc,0.975):.3f}])"
          f" | 蛋白R2中位={np.median(r2_arr):.3f} (95%CI [{blo:.3f},{bhi:.3f}])  n={len(idx)}")

print("\n=== 二、按蛋白类别分层的蛋白 R²（v52ens 集成）===")
print(f"{'场景':<18}{'高丰度':>10}{'低丰度':>10}{'高变异':>10}{'低变异':>10}{'有GO':>10}{'无GO':>10}")
for scene in SCENES:
    idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    pred, yt, m = pred_all[scene], y_log2[idx], mask[idx]
    r = lambda sel: prot_r2(pred, yt, m, sel)
    print(f"{scene:<18}{r(ab_hi):>10.3f}{r(ab_lo):>10.3f}{r(var_hi):>10.3f}{r(var_lo):>10.3f}{r(prot_has_go):>10.3f}{r(~prot_has_go):>10.3f}")

print("\n=== 三、按化合物相似度分层（val_chem_only，chem_max_sim）===")
idx = np.where(meta['split_final'].eq('val_chem_only').values & meta['role'].eq('treatment').values)[0]
pred, yt, m = pred_all['val_chem_only'], y_log2[idx], mask[idx]
yc = ctrl_all[[pos_of[s] for s in idx]]
sims = feats['chem_max_sim'][idx]
print(f"{'分层':<14}{'n':>6}{'FC PCC':>10}{'蛋白R2':>10}")
for lo, hi, name in [(0.0, 0.2, '低相似<0.2'), (0.2, 0.5, '中相似0.2-0.5'), (0.5, 1.01, '高相似>0.5')]:
    sel = (sims >= lo) & (sims < hi)
    if sel.sum() == 0:
        continue
    p, y, mm, ycc = pred[sel], yt[sel], m[sel], yc[sel]
    print(f"{name:<14}{sel.sum():>6}{fc_pcc(p, y, ycc, mm):>10.3f}{prot_r2(p, y, mm):>10.3f}")

print("\n=== 四、按菌株遗传距离分层（到最近训练菌株的 SNP 距离）===")
sdv = feats['strain_dist_vec'].astype(np.float64)
nearest = sdv.min(axis=1)
print(f"{'场景':<18}{'n(近<1.0)':>12}{'近FC':>10}{'n(远>=1.0)':>12}{'远FC':>10}")
for scene in ['val_strain_only', 'val_both']:
    idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    pred, yt, m = pred_all[scene], y_log2[idx], mask[idx]
    yc = ctrl_all[[pos_of[s] for s in idx]]
    d = nearest[idx]
    for lo, hi, name in [(0.0, 1.0, '近<1.0'), (1.0, 99.0, '远>=1.0')]:
        sel = (d >= lo) & (d < hi)
        if sel.sum() > 0:
            p, y, mm, ycc = pred[sel], yt[sel], m[sel], yc[sel]
            print(f"{scene:<18}{sel.sum():>12}{fc_pcc(p, y, ycc, mm):>10.3f}", end='')
        else:
            print(f"{scene:<18}{0:>12}{'--':>10}", end='')
    print()
