# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 01 数据预处理
对齐 -> 角色识别 -> 缺失过滤(仅训练行, <80%) -> log2 -> mask
输出到 vcell/data/
"""
import os
import numpy as np
import pandas as pd

BASE = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"
OUT = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 读取与对齐 ----------
meta = pd.read_csv(f"{BASE}/WAYB_WAYC_metadata_train_val(1).csv")
prot = pd.read_csv(f"{BASE}/WAYB_WAYC_proteome_raw_train_val.csv")
meta = meta.set_index('sample_ID')
prot = prot.set_index('sample_ID')
assert meta.index.equals(prot.index), "sample_ID 未对齐"
N = len(meta)
print(f"[对齐] 样本数 {N}，蛋白数 {prot.shape[1]}")

# ---------- 2. 角色识别 ----------
def role_of(s):
    s = str(s).strip()
    if s.upper() in ('DMSO', 'WATER'):
        return 'control'
    if 'QUALITY' in s.upper():
        return 'qc'
    return 'treatment'
meta['role'] = meta['perturbation_no_concentration'].map(role_of)
print("[角色]", meta['role'].value_counts().to_dict())

# ---------- 3. 缺失过滤（仅训练行）----------
train_mask = meta['split_final'].eq('train').values
missing_rate = prot.loc[train_mask].isna().mean(axis=0)
keep = missing_rate < 0.80
P = int(keep.sum())
print(f"[过滤] {prot.shape[1]} -> {P} 蛋白 (缺失率<80%, 仅训练行统计)")

prot_f = prot.loc[:, keep]
mask = (~prot_f.isna()).values.astype(np.float32)
y_log2 = np.log2(prot_f).values.astype(np.float32)   # NaN 保留
print(f"[log2] 值域 [{np.nanmin(y_log2):.2f}, {np.nanmax(y_log2):.2f}]")
print(f"[mask] 全数据有效值比例 {mask.mean():.3f}，训练集 {mask[train_mask].mean():.3f}")

# ---------- 4. 保存 ----------
np.save(f"{OUT}/y_log2.npy", y_log2)
np.save(f"{OUT}/mask.npy", mask)
np.save(f"{OUT}/keep_proteins.npy", keep.values)
meta.to_pickle(f"{OUT}/meta.pkl")
with open(f"{OUT}/prot_names.txt", 'w', encoding='utf-8') as f:
    f.write('\n'.join(prot_f.columns))
print(f"[保存] y_log2.npy {y_log2.shape} | mask.npy {mask.shape}")
print("01 DONE")
