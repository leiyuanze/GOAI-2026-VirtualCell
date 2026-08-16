# -*- coding: utf-8 -*-
"""自适应迁移 α 前置分析：test 新化合物 top-1 相似度分布 + val 伪新化合物上 α 敏感性"""
import numpy as np, pandas as pd, pickle

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
BASE = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"
feats = pickle.load(open(f"{DATA}/feats.pkl", 'rb'))
meta = pd.read_pickle(f"{DATA}/meta.pkl")
pert2morgan = feats['pert2morgan64']

train_chems = sorted(meta.loc[meta['split_final'].eq('train') & meta['role'].eq('treatment'), 'perturbation_no_concentration'])
tmeta = pd.read_csv(f"{BASE}/WAYB_WAYC_metadata_test(1).csv").set_index('sample_ID')
# test 无 role 列：对照 = Water/DMSO，质控 = Quality
t_is_ctrl = tmeta['perturbation_no_concentration'].isin(['Water', 'DMSO']).values
t_is_qc = tmeta['perturbation_no_concentration'].astype(str).str.contains('Quality', case=False, na=False).values
t_is_treat = (~t_is_ctrl) & (~t_is_qc)
test_chems = sorted(set(tmeta.loc[t_is_treat, 'perturbation_no_concentration']) - set(train_chems))
print(f"test 新化合物 {len(test_chems)} 个")

def top_sims(c, k=5):
    v = pert2morgan.get(c)
    if v is None:
        return []
    sims = []
    for tc in train_chems:
        w = pert2morgan.get(tc)
        if w is None:
            continue
        s = float(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-8))
        sims.append((s, tc))
    sims.sort(reverse=True)
    return sims[:k]

print(f"\n{'test 新化合物':<30}{'top1_sim':>9}{'top3_avg':>9}{'最近邻'}")
for c in test_chems:
    tops = top_sims(c)
    if not tops:
        print(f"{c:<30}{'无指纹':>9}")
        continue
    t3 = np.mean([s for s, _ in tops[:3]])
    print(f"{c:<30}{tops[0][0]:>9.3f}{t3:>9.3f}  {tops[0][1][:20]}")

# val 伪新化合物（val_chem_only 中 train 无 Δ 的化合物）
val_chems = sorted(meta.loc[meta['split_final'].eq('val_chem_only') & meta['role'].eq('treatment'), 'perturbation_no_concentration'])
val_new = [c for c in val_chems if c not in set(train_chems)]
print(f"\nval 伪新化合物 {len(val_new)} 个（可用于 α 敏感性验证）:")
for c in val_new:
    tops = top_sims(c)
    if tops:
        print(f"  {c:<30} top1={tops[0][0]:.3f}  {tops[0][1][:20]}")
