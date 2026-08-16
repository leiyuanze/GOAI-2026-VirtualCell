# -*- coding: utf-8 -*-
"""
分布偏移审计 v2（gpt3 §二.3 / opus1 阶段一）：
核心对比：val_chem vs test_chem_only（新化合物）、val_strain/both vs test_strain_only/test_both（新菌株）
特征：化学最大相似度、遗传最近距离、时间点占比、培养基占比、平台占比、缺失率
"""
import pickle
import numpy as np
import pandas as pd
import gzip

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
INPUT = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"

meta = pd.read_pickle(f"{DATA}/meta.pkl")
mask = np.load(f"{DATA}/mask.npy").astype(np.float32)
with open(f"{DATA}/feats.pkl", 'rb') as f:
    feats = pickle.load(f)
tmeta = pd.read_csv(f"{INPUT}/WAYB_WAYC_metadata_test(1).csv").set_index('sample_ID')
traw = pd.read_csv(f"{INPUT}/WAYB_WAYC_proteome_raw_test.csv").set_index('sample_ID')

train_mask = meta['split_final'].eq('train').values

# ---------- 化学最大相似度（到训练化合物） ----------
pert2morgan = feats['pert2morgan64']
train_chems = sorted(set(meta.loc[train_mask & meta['role'].eq('treatment'), 'perturbation_no_concentration']))
train_morgan = np.stack([pert2morgan[c] for c in train_chems])
train_morgan = train_morgan / (np.linalg.norm(train_morgan, axis=1, keepdims=True) + 1e-8)

def max_chem_sim(chem):
    v = pert2morgan.get(chem)
    if v is None or np.abs(v).sum() == 0:
        return np.nan
    v = v / (np.linalg.norm(v) + 1e-8)
    return float((train_morgan @ v).max())

# ---------- 遗传最近距离（菌株→最近训练菌株；test 菌株从 SNP 矩阵直接算） ----------
train_strains = sorted(set(meta.loc[train_mask, 'Strains']))
with gzip.open(f"{DATA}/1011_SNP_distance.tab.gz", 'rt') as f:
    names = f.readline().strip().split('\t')
    dist = np.loadtxt(f, usecols=range(1, len(names)))
name2i = {n: i for i, n in enumerate(names)}

def strain_nearest_dist(s):
    # 到最近训练菌株的 SNP 距离
    vals = []
    for ts in train_strains:
        if s in name2i and ts in name2i:
            vals.append(dist[name2i[s], name2i[ts]])
    if s in train_strains:
        return 0.0
    return float(min(vals)) if vals else float('nan')

# ---------- 组别 ----------
G = {}
G['train'] = meta[train_mask]
for sc in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
    G[sc] = meta[meta['split_final'].eq(sc)]
for sc in ['test_chem_only', 'test_strain_only', 'test_both', 'test_time']:
    G[sc] = tmeta[tmeta['split_final'].eq(sc)]

ORDER = ['train', 'val_chem_only', 'test_chem_only', 'val_strain_only', 'test_strain_only',
         'val_both', 'test_both', 'val_time', 'test_time']
HDR = {'train': 'train', 'val_chem_only': 'val_chem', 'test_chem_only': 'test_chem',
       'val_strain_only': 'val_strain', 'test_strain_only': 'test_strain',
       'val_both': 'val_both', 'test_both': 'test_both',
       'val_time': 'val_time', 'test_time': 'test_time'}

def chem_sims(g):
    return [max_chem_sim(c) for c in g['perturbation_no_concentration'].unique() if not pd.isna(c)]

def time_share(g, t):
    vals = g['pert_time'].astype(float).values
    return float((np.abs(vals - t) < 1e-6).mean())

def med_share(g, m):
    return float((g['Medium'].astype(str) == m).mean())

def plat_share(g, col):
    return float(g[col].astype(str).value_counts(normalize=True).iloc[0])

def missing_rate(g):
    if 'split_final' in g.columns:
        return float((~mask.astype(bool)).mean())
    vals = traw.loc[g.index]
    return float((~np.isfinite(vals.values)).mean())

print("\n=== 分布偏移审计表（关键对比：新化合物 / 新菌株） ===")
hdr_str = f"{'特征':<26}" + "".join(f"{HDR[k]:>11}" for k in ORDER)
print(hdr_str)

def row(name, fn, fmt='.3f'):
    vals = [fn(G[k]) for k in ORDER]
    s = f"{name:<26}" + "".join((f"{v:>11.3f}" if isinstance(v, float) else f"{v:>11}") for v in vals)
    print(s)
    return vals

row('化合物数', lambda g: len(chem_sims(g)), 'd')
row('化学最大相似度(中位)', lambda g: float(np.nanmedian(chem_sims(g))) if len(chem_sims(g)) else float('nan'))
row('化学最大相似度(均值)', lambda g: float(np.nanmean(chem_sims(g))) if len(chem_sims(g)) else float('nan'))
row('遗传最近距离(菌株级)', lambda g: float(np.nanmedian([strain_nearest_dist(s) for s in g['Strains'].unique()])) if 'Strains' in g.columns else float('nan'))
row('15min 占比', lambda g: time_share(g, 15.0))
row('240min 占比', lambda g: time_share(g, 240.0))
row('galactose 培养基占比', lambda g: med_share(g, 'YNB+CSM+2% galactose'))
row('glucose 培养基占比', lambda g: med_share(g, 'YNB+CSM+2% glucose'))
row('data_source 最大占比', lambda g: plat_share(g, 'data_source'))
row('instrument 最大占比', lambda g: plat_share(g, 'instrument'))
row('蛋白缺失率', missing_rate)

print("\n=== 判断 ===")
cs_val = np.nanmedian(chem_sims(G['val_chem_only']))
cs_test = np.nanmedian(chem_sims(G['test_chem_only']))
gd_val = np.nanmedian([strain_nearest_dist(s) for s in G['val_strain_only']['Strains'].unique()])
gd_test = np.nanmedian([strain_nearest_dist(s) for s in G['test_strain_only']['Strains'].unique()])
print(f"化学：val_chem 中位相似度 {cs_val:.3f} vs test_chem {cs_test:.3f}"
      f" -> {'测试化合物更远（val 乐观）' if cs_test < cs_val - 0.05 else ('测试化合物更近（val 悲观）' if cs_test > cs_val + 0.05 else '接近')}")
print(f"遗传：val_strain 最近距离 {gd_val:.3f} vs test_strain {gd_test:.3f}"
      f" -> {'测试菌株更远' if gd_test > gd_val + 0.1 else ('测试菌株更近' if gd_test < gd_val - 0.1 else '接近')}")
