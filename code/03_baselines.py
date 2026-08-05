# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 03 双基线锚点（v2，对齐官方诊断表 4.2.2 口径）
两个基线均在「能找到 exact matched control 且真值非缺失」的处理样本子集上评估
"""
import numpy as np
import pandas as pd

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy")
mask = np.load(f"{DATA}/mask.npy").astype(bool)
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values

# ---------- Matched Control 查找表 ----------
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
    if k not in ctrl_lookup:
        return None
    rows = ctrl_lookup[k]
    cvals = np.where(mask, y_log2, np.nan)[rows]
    cm = mask[rows]
    with np.errstate(invalid='ignore'):
        return np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)

# ---------- 向量化指标 ----------
def masked_rmse(yt, yp, m):
    return float(np.sqrt(((yt[m] - yp[m]) ** 2).mean()))

def masked_global_r2(yt, yp, m):
    a, b = yt[m], yp[m]
    ss_res = ((a - b) ** 2).sum()
    ss_tot = ((a - a.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

def protein_r2_median(yt, yp, m):
    cnt = m.sum(0)
    keep = cnt >= 3
    n = np.maximum(cnt.astype(float), 1)
    yt_c = np.where(m, yt, 0.0); yp_c = np.where(m, yp, 0.0)
    mt = yt_c.sum(0) / n
    ss_tot = (((yt_c - mt) ** 2) * m).sum(0)      # 仅在观测位置
    ss_res = (((yt_c - yp_c) ** 2) * m).sum(0)    # 仅在观测位置
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    return float(np.median(r2[keep])) if keep.any() else float('nan')

def evaluate(name, yp_dict, sub):
    print(f"\n===== {name} =====", flush=True)
    print(f"{'场景':<16}{'样本数':>6}{'log2RMSE':>10}{'GlobalR2':>10}{'蛋白R2中位':>10}", flush=True)
    for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
        v = sub[scene]
        n = int(v.sum())
        if n == 0:
            print(f"{scene:<16}{0:>6}{'--':>10}{'--':>10}{'--':>10}", flush=True)
            continue
        idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
        yt, m = y_log2[idx][v], mask[idx][v]
        yp = yp_dict[scene][v]
        m = m & np.isfinite(yp)          # 对照缺失位置屏蔽
        rmse = masked_rmse(yt, yp, m)
        g2 = masked_global_r2(yt, yp, m)
        p2 = protein_r2_median(yt, yp, m)
        print(f"{scene:<16}{n:>6}{rmse:>10.3f}{g2:>10.3f}{p2:>10.3f}", flush=True)

# ---------- 构建各场景子集（找到 matched control 且真值非全缺失）----------
sub = {}
yp_ctrl = {}
for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
    s = meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values
    idx = np.where(s)[0]
    yp = np.full((len(idx), P), np.nan)
    found = np.zeros(len(idx), dtype=bool)
    for i, sid in enumerate(idx):
        cm = matched_control_mean(sid)
        if cm is not None:
            yp[i] = cm
            found[i] = True
    has_obs = mask[idx].sum(axis=1) > 0
    sub[scene] = found & has_obs
    yp_ctrl[scene] = yp
    print(f"[子集] {scene}: 处理样本 {len(idx)} -> 有效 {int((found & has_obs).sum())}", flush=True)

# ---------- 基线一：蛋白均值（同一子集评估）----------
protein_mean = np.nanmean(np.where(mask, y_log2, np.nan)[train_mask], axis=0)
yp_mean = {sc: np.tile(protein_mean, (len(yp_ctrl[sc]), 1)) for sc in yp_ctrl}
evaluate("基线一：蛋白均值", yp_mean, sub)

# ---------- 基线二：Matched Control ----------
evaluate("基线二：Matched Control", yp_ctrl, sub)

print("\n===== 官方诊断表（教程 4.2.2，供对标）=====", flush=True)
official = [
    ('双重未知', '蛋白均值', 266, 0.994, 0.871, -0.064),
    ('双重未知', 'Matched control', 266, 0.382, 0.98, 0.809),
    ('新化合物', '蛋白均值', 1015, 1.004, 0.868, -0.038),
    ('新化合物', 'Matched control', 1015, 0.379, 0.98, 0.836),
    ('新菌株', '蛋白均值', 1293, 0.875, 0.897, -0.036),
    ('新菌株', 'Matched control', 1293, 0.399, 0.978, 0.726),
    ('时间验证', '蛋白均值', 128, 0.869, 0.9, -0.009),
    ('时间验证', 'Matched control', 128, 0.426, 0.975, 0.719),
]
print(f"{'场景':<8}{'基线':<18}{'样本数':>6}{'log2RMSE':>10}{'GlobalR2':>10}{'蛋白R2中位':>10}", flush=True)
for r in official:
    print(f"{r[0]:<8}{r[1]:<18}{r[2]:>6}{r[3]:>10.3f}{r[4]:>10.3f}{r[5]:>10.3f}", flush=True)
print("03 DONE", flush=True)
