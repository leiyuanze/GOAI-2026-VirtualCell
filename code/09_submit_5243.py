# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 09 提交列补齐（4422 -> 5243）
官方 feature contract 要求提交全部 5243 个蛋白列（与 proteome 文件一致）。
模型只输出 4422 个保留蛋白（缺失率<80%），本脚本把 821 个被过滤蛋白补回：
填充值 = 该蛋白在 train 划分的 log2 均值（完全无观测则用全局中位数兜底）。
仅用 train 划分计算统计量，不触碰 val/test 标签（合规）。
"""
import numpy as np
import pandas as pd

BASE = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"
DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
SRC = "prediction_ensemble6.csv"   # 07e 生成，4454 x 4422
DST = "prediction_final_5243.csv"  # 最终提交，4454 x 5243

# 官方蛋白列顺序（test 与 train_val 一致）
test_cols = pd.read_csv(f"{BASE}/WAYB_WAYC_proteome_raw_test.csv", nrows=0).columns.tolist()
prot_order = test_cols[1:]
assert len(prot_order) == 5243, f"官方蛋白列数异常: {len(prot_order)}"

old = pd.read_csv(f"{DATA}/{SRC}", index_col=0)

# 训练集 log2 蛋白均值（仅 train 划分）
meta = pd.read_csv(f"{BASE}/WAYB_WAYC_metadata_train_val(1).csv")
train_ids = meta.loc[meta['split_final'] == 'train', 'sample_ID']
prot = pd.read_csv(f"{BASE}/WAYB_WAYC_proteome_raw_train_val.csv", index_col='sample_ID')
prot_train = prot.loc[train_ids]
log2_mean = np.log2(prot_train).mean(axis=0, skipna=True)
log2_mean = log2_mean.fillna(np.median(log2_mean.dropna()))

new = pd.DataFrame(np.nan, index=old.index, columns=prot_order)
kept = [c for c in prot_order if c in old.columns]
new[kept] = old[kept]
for c in prot_order:
    if c not in old.columns:
        new[c] = log2_mean[c]

assert new.shape == (old.shape[0], 5243)
assert not new.isna().any().any() and np.isfinite(new.values).all()
new.to_csv(f"{DATA}/{DST}")
print(f"09 DONE: {DST}  {new.shape}")
