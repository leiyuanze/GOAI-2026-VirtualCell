# -*- coding: utf-8 -*-
"""用 _test_score 相同逻辑评估 val 集，验证口径正确性"""
import numpy as np, pandas as pd, sys

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
SUBMIT = sys.argv[1] if len(sys.argv) > 1 else f'{DATA}/prediction_final_5243.csv'

meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float64)
mask = np.load(f'{DATA}/mask.npy').astype(bool)
P = y_log2.shape[1]
train_mask = meta['split_final'].eq('train').values
tr_y_nan = np.where(mask, y_log2, np.nan)

# 提交（5243 列）—— 需要从 5243 映射回 4422
cols5243 = list(pd.read_csv(SUBMIT, nrows=1).columns[1:])
prot4422 = [l.strip() for l in open(f'{DATA}/prot_names.txt')]
col_of = {p: i for i, p in enumerate(cols5243)}
pos4422 = np.array([col_of[p] for p in prot4422], dtype=int)

# 对 val 用 4422 的模型预测更合理：直接用 y_log2 的 P=4422 列
# 简化：评估提交的 4422 子集
sub = pd.read_csv(SUBMIT, index_col=0)
# meta 的顺序应与 y_log2 行一致
assert len(meta) == len(y_log2)

def mk_key(df):
    return (df['data_source'].astype(str) + '|' + df['instrument'].astype(str) + '|'
            + df['Yeast_cell_plate'].astype(str) + '|' + df['Strains'].astype(str) + '|'
            + df['Medium'].astype(str) + '|' + df['Temperature'].astype(str) + '|'
            + df['pert_time'].astype(str)).values

# 训练对照池（4422 蛋白）
ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = mk_key(meta)
ctrl_lookup = {}
for k, pos in zip(ctrl_key, ctrl_idx):
    ctrl_lookup.setdefault(k, []).append(pos)

# val 处理样本的 matched control
scenes = ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']
all_treat = np.where(meta['role'].eq('treatment').values)[0]
ctrl_vals = np.full((len(all_treat), P), np.nan)
for i, sid in enumerate(all_treat):
    rows = ctrl_lookup.get(ctrl_key[sid], [])
    if not rows:
        continue
    cvals = tr_y_nan[rows]
    cm = np.isfinite(cvals)
    with np.errstate(invalid='ignore'):
        ctrl_vals[i] = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, 0), 0) / cm.sum(0), np.nan)
pos_of = {sid: i for i, sid in enumerate(all_treat)}

# 训练集 μ_ctx / μ_drug（LOO 等价：用训练行均值）
chem_of = meta['perturbation_no_concentration'].values[all_treat]
ctx_key = (meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|'
           + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)).values[all_treat]
is_train_treat = np.isin(all_treat, np.where(train_mask & meta['role'].eq('treatment').values)[0])
tr_treat_idx = np.where(is_train_treat)[0]
delta_tr = tr_y_nan[all_treat] - ctrl_vals
mu_ctx_pool = {}; mu_drug_pool = {}
for key, members in pd.Series(tr_treat_idx, index=ctx_key[is_train_treat]).groupby(level=0):
    mu_ctx_pool[key] = np.nanmean(delta_tr[members.values], axis=0)
for key, members in pd.Series(tr_treat_idx, index=chem_of[is_train_treat]).groupby(level=0):
    mu_drug_pool[key] = np.nanmean(delta_tr[members.values], axis=0)

def safe_pcc(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10: return 0.0
    a, b = a[ok], b[ok]
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0

def safe_r2(yp, yt, ok):
    if ok.sum() < 3: return 0.0
    a, b = yp[ok], yt[ok]
    ss_res = ((a - b) ** 2).sum(); ss_tot = ((b - b.mean()) ** 2).sum()
    if ss_tot < 1e-12: return 0.0
    r2 = 1 - ss_res / ss_tot
    return float(r2) if np.isfinite(r2) else 0.0

print(f"{'场景':<16}{'M1:FC':>8}{'M2:绝对':>8}{'M3:ctx':>8}{'M4:drug':>8}{'M6:DEP':>8}{'样本':>6}")
print('-' * 55)
for scene in scenes:
    idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    if len(idx) == 0:
        continue
    # 提交预测（4422 蛋白子集），对齐 meta 行
    # 注意：提交文件是 test 的，val 无提交。这里用模型预测——直接要求传入 val 预测文件
    print(f"{scene}: 需要 val 预测文件，跳过")
    break
