# -*- coding: utf-8 -*-
"""
P0 泄漏审计（按 gpt1/gpt2 指示）
1. ctx_prior self-inclusion：训练样本的 (strain|medium|temp|time) 分组均值是否包含样本自身
2. matched control 跨划分泄漏：训练处理样本的对照是否取到 val/test 划分的对照行
3. 统计量 train-only 合规性核对
4. ctx_prior 分组中处理样本占比（影响 M3/M4 语义）
"""
import os, numpy as np, pandas as pd

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy")
mask = np.load(f"{DATA}/mask.npy")
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values
tr_y = np.where(mask.astype(bool), y_log2, np.nan)

print("=" * 70)
print("审计 1: ctx_prior self-inclusion（训练样本分组均值是否含自身）")
print("=" * 70)
ctx_key = meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|' + \
          meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)
df_ctx = pd.DataFrame({'key': ctx_key.values, 'tr': train_mask, 'role': meta['role'].values})
# 分组内样本数分布（训练行）
grp_sizes = df_ctx[df_ctx['tr']].groupby('key').size()
print(f"训练行 ctx 分组数: {len(grp_sizes)} | 分组大小 中位 {grp_sizes.median():.0f} 最大 {grp_sizes.max()} 单例组 {int((grp_sizes==1).sum())}")

# 若分组大小=1，则 ctx_prior = 样本自身真值（完全 self-inclusion）
single_self = grp_sizes[grp_sizes == 1]
print(f"⚠️ 单样本组（ctx_prior = 自身真值，完全泄漏）: {len(single_self)} 组, 覆盖训练样本 {int((df_ctx['tr'] & df_ctx['key'].isin(single_self.index)).sum())} 个")

# 处理样本的 ctx_prior 中，处理行 vs 对照行占比
tr_df = df_ctx[df_ctx['tr']]
tr_df = tr_df.copy()
tr_df['n'] = 1
pivot = tr_df.groupby(['key', 'role']).size().unstack(fill_value=0)
if 'control' in pivot.columns and 'treatment' in pivot.columns:
    treat_share = pivot['treatment'] / (pivot['treatment'] + pivot['control'] + pivot.get('qc', 0))
    print(f"\n处理样本在 ctx 分组均值中的占比: 中位 {treat_share.median():.2f} | "
          f"纯处理组 {int((treat_share == 1).sum())} 个 | 含对照组 {int((treat_share < 1).sum())} 个")
    print("⚠️ ctx_prior 混合处理/对照：处理样本的上下文先验包含同条件其他处理的响应均值")

print("\n" + "=" * 70)
print("审计 2: matched control 跨划分泄漏（训练处理样本的对照来自哪个划分）")
print("=" * 70)
ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = (meta.iloc[ctrl_idx]['data_source'].astype(str) + '|' + meta.iloc[ctrl_idx]['instrument'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Yeast_cell_plate'].astype(str) + '|' + meta.iloc[ctrl_idx]['Strains'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['Medium'].astype(str) + '|' + meta.iloc[ctrl_idx]['Temperature'].astype(str) + '|'
            + meta.iloc[ctrl_idx]['pert_time'].astype(str)).values
ctrl_split = meta['split_final'].values[ctrl_idx]

from collections import defaultdict
lookup = defaultdict(list)
for k, pos, sp in zip(ctrl_key, ctrl_idx, ctrl_split):
    lookup[k].append((pos, sp))

# 训练处理样本
train_treat = np.where(train_mask & meta['role'].eq('treatment').values)[0]
cross = 0; same_train = 0; no_ctrl = 0
cross_samples = []
for sid in train_treat:
    r = meta.iloc[sid]
    k = (str(r['data_source']) + '|' + str(r['instrument']) + '|' + str(r['Yeast_cell_plate']) + '|'
         + str(r['Strains']) + '|' + str(r['Medium']) + '|' + str(r['Temperature']) + '|' + str(r['pert_time']))
    if k not in lookup:
        no_ctrl += 1
        continue
    splits = {sp for _, sp in lookup[k]}
    if 'train' in splits:
        same_train += 1
    if splits - {'train'}:
        cross += 1
        cross_samples.append((sid, r['Strains'], r['perturbation_no_concentration'], sorted(splits)))
print(f"训练处理样本 {len(train_treat)}: 有同键训练对照 {same_train} | 键同时命中 val/test 对照 {cross} | 无对照 {no_ctrl}")
if cross_samples:
    print("跨划分样本示例 (sid, strain, chem, 命中的划分):")
    for s in cross_samples[:10]:
        print("  ", s)
    # 命中 val 对照的具体数量
    val_only = sum(1 for _, _, _, sp in cross_samples if sp == ['val'] or sp == ['test'] or (len(sp) == 1 and sp[0] != 'train'))
    print(f"⚠️ 其中仅命中非 train 划分（val/test）的对照: {val_only} 个 —— 训练监督用了 val 真值")
else:
    print("✅ 无跨划分泄漏：训练处理样本的对照全部来自 train 划分")

print("\n" + "=" * 70)
print("审计 3: 统计量 train-only 核对（02_features 关键统计量来源）")
print("=" * 70)
checks = [
    ("gmean（蛋白全局均值）", "np.nanmean(tr_y[train_mask])  ✅ 仅训练"),
    ("strain_means（菌株均值）", "rows = train_mask & (strains==s)  ✅ 仅训练"),
    ("chem_delta_mean（化合物Δ均值）", "matched_control + treat 行，未过滤 split_final ⚠️ 需核对"),
    ("ctx_prior（上下文先验）", "df_ctx[df_ctx['tr']] 仅训练行  ✅ 但分组含自身 ⚠️"),
    ("共表达相关矩阵", "X = tr_y[train_mask]  ✅ 仅训练"),
    ("SVD 谱聚类", "基于共表达（训练）✅"),
]
for name, status in checks:
    print(f"  {name}: {status}")

# 化合物 Δ 均值是否含 val 处理行
chem_delta_used = np.where(meta['role'].eq('treatment').values)[0]
from_split = meta['split_final'].values[chem_delta_used]
print(f"\nchem_delta_mean 用到的处理行（全部划分）: {len(chem_delta_used)} | "
      f"其中非 train: {int((from_split != 'train').sum())} 个"
      f"（val {int((from_split == 'val_chem_only').sum() + (from_split == 'val_strain_only').sum() + (from_split == 'val_both').sum() + (from_split == 'val_time').sum())} / test {int((from_split == 'test_chem_only').sum() + (from_split == 'test_strain_only').sum() + (from_split == 'test_both').sum() + (from_split == 'test_time').sum())}）")

print("\n审计完成")
