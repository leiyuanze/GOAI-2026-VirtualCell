# -*- coding: utf-8 -*-
"""公平对比：所有模型 + 基线在同一评估标准下"""
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

# matched control lookup
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
    if k not in ctrl_lookup: return None
    rows = ctrl_lookup[k]; cvals = tr_y_nan[rows]; cm = mask[rows] > 0
    with np.errstate(invalid='ignore'):
        return np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)

train_mask = meta['split_final'].eq('train').values
protein_mean = np.nanmean(tr_y_nan[train_mask], axis=0)

def prot_r2(yp, yt, m):
    cnt = m.sum(0); keep = cnt >= 3; n = np.maximum(cnt.astype(float), 1)
    ytc = np.where(m, yt, 0.0); ypc = np.where(m, yp, 0.0)
    mt = ytc.sum(0) / n
    ss_tot = (((ytc - mt) ** 2) * m).sum(0); ss_res = (((ytc - ypc) ** 2) * m).sum(0)
    return float(np.median(1 - ss_res / np.maximum(ss_tot, 1e-12)))

def fc_pcc(yp, yt, m, yc):
    ok = np.isfinite(yc) & m & np.isfinite(yp)
    if ok.sum() < 10: return float('nan')
    dp = (yp - yc)[ok]; dt = (yt - yc)[ok]
    return float(np.corrcoef(dp, dt)[0, 1])

# Load models
_s21 = importlib.util.spec_from_file_location('m21', r'D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v21.py')
_m21 = importlib.util.module_from_spec(_s21); _s21.loader.exec_module(_m21)
_s27 = importlib.util.spec_from_file_location('m27', r'D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v27.py')
_m27 = importlib.util.module_from_spec(_s27); _s27.loader.exec_module(_m27)

def load(path, cls, set_avg=False):
    m = cls(feats, P=P); m.load_state_dict(torch.load(f'{DATA}/{path}', map_location=DEV, weights_only=True))
    if set_avg: m.set_strain_avg()
    return m.to(DEV).eval()

m21 = load('model_v21.pt', _m21.VCellModel)
m21s43 = load('model_v21_s43.pt', _m21.VCellModel)
m21s44 = load('model_v21_s44.pt', _m21.VCellModel)
m27 = load('model_v27_best.pt', _m27.VCellModel, set_avg=True)

def predict(idx, models_list):
    preds = []
    for m, tag in models_list:
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

# ============================================================
# 对比表 1: 蛋白R2中位数（全样本 vs 基线子集）
# ============================================================
print("=" * 110)
print("表1: 逐蛋白 R2 中位数对比（蛋白R2中位）")
print("-" * 110)
hdr = f"{'场景':<18}{'全样本':>6}{'有MC子集':>8}{'蛋白均值':>9}{'MC基线':>9}{'v2.1单':>9}{'v2.7单':>9}{'新集成':>9}"
print(hdr)
print("-" * 110)

for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
    idx_all = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]

    yc_arr = np.array([matched_control_mean(s) for s in idx_all])
    has_mc = np.array([y is not None and np.isfinite(y).any() for y in yc_arr])
    has_obs = mask[idx_all].sum(axis=1) > 0
    sub_idx = idx_all[has_mc & has_obs]
    if len(sub_idx) == 0: continue

    yt_sub = y_log2[sub_idx]; m_sub = mask[sub_idx].astype(bool)
    yt_all = y_log2[idx_all]; m_all = mask[idx_all].astype(bool)
    yc_all = np.array([matched_control_mean(s) for s in idx_all])

    # 蛋白均值（子集）
    yp_mean = np.tile(protein_mean, (len(sub_idx), 1))
    r2_mean = prot_r2(np.where(m_sub, yp_mean, 0.0), np.where(m_sub, yt_sub, 0.0), m_sub)

    # MC基线（子集）
    yc_sub = np.array([matched_control_mean(s) for s in sub_idx])
    yc_sub = np.nan_to_num(yc_sub, nan=0.0)
    r2_mc = prot_r2(np.where(m_sub, yc_sub, 0.0), np.where(m_sub, yt_sub, 0.0), m_sub)

    # v2.1 单（全样本）
    yp21 = predict(idx_all, [(m21, 'v21')])
    r2_21 = prot_r2(np.where(m_all, yp21, 0.0), np.where(m_all, yt_all, 0.0), m_all)

    # v2.7 单（全样本）
    yp27 = predict(idx_all, [(m27, 'v27')])
    r2_27 = prot_r2(np.where(m_all, yp27, 0.0), np.where(m_all, yt_all, 0.0), m_all)

    # 新集成（全样本）
    yp_ens = predict(idx_all, [(m21,'v21'),(m21s43,'v21'),(m21s44,'v21'),(m27,'v27')])
    r2_ens = prot_r2(np.where(m_all, yp_ens, 0.0), np.where(m_all, yt_all, 0.0), m_all)

    n_all = len(idx_all); n_sub = len(sub_idx)
    line = f"{scene:<18}{n_all:>6}{n_sub:>8}{r2_mean:>9.3f}{r2_mc:>9.3f}{r2_21:>9.3f}{r2_27:>9.3f}{r2_ens:>9.3f}"
    print(line)

# ============================================================
# 对比表 2: FC PCC（扰动效应预测，25%权重核心指标）
# ============================================================
print()
print("=" * 110)
print("表2: Fold Change PCC 对比（匹配对照原始FC，25%权重）")
print("-" * 110)
hdr2 = f"{'场景':<18}{'蛋白均值':>9}{'MC基线':>9}{'v2.1单':>9}{'v2.7单':>9}{'新集成':>9}"
print(hdr2)
print("-" * 110)

for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
    idx_all = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    yt_all = y_log2[idx_all]; m_all = mask[idx_all].astype(bool)
    yc_all = np.array([matched_control_mean(s) for s in idx_all])

    # 蛋白均值
    yp_mean = np.tile(protein_mean, (len(idx_all), 1))
    fc_mean = fc_pcc(np.where(m_all, yp_mean, 0.0), yt_all, m_all, yc_all)

    # MC 基线
    yp_mc = np.nan_to_num(np.array([matched_control_mean(s) for s in idx_all]), nan=0.0)
    fc_mc = fc_pcc(yp_mc, yt_all, m_all, yc_all)

    # v2.1
    yp21 = predict(idx_all, [(m21, 'v21')])
    fc_21 = fc_pcc(yp21, yt_all, m_all, yc_all)

    # v2.7
    yp27 = predict(idx_all, [(m27, 'v27')])
    fc_27 = fc_pcc(yp27, yt_all, m_all, yc_all)

    # 新集成
    yp_ens = predict(idx_all, [(m21,'v21'),(m21s43,'v21'),(m21s44,'v21'),(m27,'v27')])
    fc_ens = fc_pcc(yp_ens, yt_all, m_all, yc_all)

    line2 = f"{scene:<18}{fc_mean:>9.3f}{fc_mc:>9.3f}{fc_21:>9.3f}{fc_27:>9.3f}{fc_ens:>9.3f}"
    print(line2)

print()
print("=" * 110)
print("解读:")
print("  蛋白R2 = 逐蛋白跨样本的R2中位数, 衡量\"每个蛋白的绝对丰度预测有多准\"")
print("  FC PCC = PCC(Delta_pred, Delta_true), 衡量\"预测的扰动方向/幅度是否对\"")
print("  蛋白均值: 对所有样本输出训练集蛋白均值向量")
print("  MC基线: 输出同条件matched control的蛋白值（不建模的上限）")
print("  模型: 输出模型预测的绝对丰度")
print()
print("  基线子集 < 全样本: 部分处理样本找不到完全匹配的对照")
print("  蛋白均值/MC基线需要对照值，所以只在子集上评估")
print("  模型可以预测所有样本，在全样本上评估更严格")
print("=" * 110)
