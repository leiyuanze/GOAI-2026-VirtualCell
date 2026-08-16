# -*- coding: utf-8 -*-
"""opus4 case study：选一个 test 新化合物，预测 Δ vs 真值 Δ + top-k 差异蛋白 GO 富集"""
import numpy as np
import pandas as pd
import pickle

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
INPUT = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"

meta = pd.read_pickle(f"{DATA}/meta.pkl")
y_log2 = np.load(f"{DATA}/y_log2.npy").astype(np.float64)
mask = np.load(f"{DATA}/mask.npy").astype(bool)
with open(f"{DATA}/feats.pkl", 'rb') as f:
    feats = pickle.load(f)
P = y_log2.shape[1]
tr_y_nan = np.where(mask, y_log2, np.nan)
train_mask = meta['split_final'].eq('train').values

tmeta = pd.read_csv(f"{INPUT}/WAYB_WAYC_metadata_test(1).csv").set_index('sample_ID')
traw = pd.read_csv(f"{INPUT}/WAYB_WAYC_proteome_raw_test.csv").set_index('sample_ID')
sub = pd.read_csv(f"{DATA}/prediction_final_0816.csv", index_col=0)
cols = sub.columns.tolist()
t_log2 = np.log2(traw[cols].values.astype(np.float64))
prot4422 = [l.strip() for l in open(f"{DATA}/prot_names.txt")]
col_of = {p: i for i, p in enumerate(cols)}
pos4422 = np.array([col_of[p] for p in prot4422], dtype=int)

# 对照（train 对照池，与 _test_score --ctrl train 一致）
ctrl_idx = np.where(meta['role'].eq('control').values & train_mask)[0]
def mk_key(df):
    return (df['data_source'].astype(str) + '|' + df['instrument'].astype(str) + '|'
            + df['Yeast_cell_plate'].astype(str) + '|' + df['Strains'].astype(str) + '|'
            + df['Medium'].astype(str) + '|' + df['Temperature'].astype(str) + '|'
            + df['pert_time'].astype(str)).values
ctrl_lookup = {}
for k, pos in zip(mk_key(meta.iloc[ctrl_idx]), ctrl_idx):
    ctrl_lookup.setdefault(k, []).append(pos)

def ctrl_of(pos):
    k = mk_key(tmeta.iloc[[pos]])[0]
    rows = ctrl_lookup.get(k, [])
    if not rows:
        return None
    cvals = tr_y_nan[rows]; cm = mask[rows] > 0
    full = np.full(len(cols), np.nan)
    with np.errstate(invalid='ignore'):
        m4422 = np.where(cm.sum(0) > 0, np.nansum(np.where(cm, cvals, np.nan), 0) / cm.sum(0), np.nan)
    full[pos4422] = m4422
    return full

# 选化合物：test_chem_only 中相似度最高的新化合物（代表"较容易"）+ 一个中等
chem_sim = feats['test_chem_max_sim']
treat_mask = ~tmeta['perturbation_no_concentration'].isin(['Water', 'DMSO']) \
             & ~tmeta['perturbation_no_concentration'].astype(str).str.contains('Quality', case=False, na=False)
chem_only = tmeta[tmeta['split_final'].eq('test_chem_only') & treat_mask]
cands = sorted(chem_only['perturbation_no_concentration'].unique())
print("test_chem_only 新化合物:", cands)
for c in cands:
    idx = chem_only.index[chem_only['perturbation_no_concentration'] == c]
    pos = tmeta.index.get_indexer(idx)
    print(f"  {c}: n={len(idx)}, max_sim={np.nanmean(chem_sim[pos]):.3f}")

# 选 2 个：最高相似度 + 中等相似度
sim_med = {c: float(np.nanmean(chem_sim[tmeta.index.get_indexer(chem_only.index[chem_only['perturbation_no_concentration'] == c])])) for c in cands}
chosen = sorted(sim_med, key=sim_med.get, reverse=True)[:1] + [sorted(sim_med, key=sim_med.get)[len(sim_med)//2]]

# GO 注释（蛋白 → 通路）
go_data = pd.read_csv(f"{DATA}/uniprot_go.tsv", sep='\t', header=None, names=['prot', 'go', 'name'])
go_of = {}
for _, row in go_data.iterrows():
    go_of.setdefault(row['prot'], []).append(row['name'])

def fisher_enrich(sel_prots, bg_prots, go_of):
    """sel 相对 bg 的 GO 富集（简单 Fisher 近似，返回 top 通路）"""
    from math import log
    sel = set(sel_prots); bg = set(bg_prots)
    N = len(bg); n = len(sel)
    results = []
    for prot in sel:
        for g in go_of.get(prot, []):
            results.append(g)
    # 通路频率
    from collections import Counter
    sel_cnt = Counter(results)
    bg_cnt = Counter()
    for prot in bg:
        for g in go_of.get(prot, []):
            bg_cnt[g] += 1
    enrich = []
    for g, cnt in sel_cnt.items():
        bg_f = bg_cnt.get(g, 0) / max(N, 1)
        sel_f = cnt / max(n, 1)
        if sel_f <= 0:
            continue
        fc = sel_f / max(bg_f, 1e-6)
        if fc >= 2 and cnt >= 2:
            enrich.append((g, cnt, round(fc, 1), round(sel_f, 3)))
    enrich.sort(key=lambda x: -x[1])
    return enrich[:6]

# 逐化合物分析
print("\n=== case study：未见化合物响应预测 ===")
for c in chosen:
    idx_ids = chem_only.index[chem_only['perturbation_no_concentration'] == c].values
    idx = tmeta.index.get_indexer(idx_ids)
    yt = t_log2[idx][:, pos4422]
    yp = sub.iloc[idx].values[:, pos4422].astype(np.float64)
    yc = np.stack([ctrl_of(i) for i in idx])[:, pos4422]
    ok = np.isfinite(yc) & np.isfinite(yt) & np.isfinite(yp)
    dt = (yt - yc)[ok]; dp = (yp - yc)[ok]
    fc = np.corrcoef(dp.ravel(), dt.ravel())[0, 1]
    # top-k 高效应蛋白（|Δ_true| 排序），预测命中率
    k = 50
    mag_true = np.abs(dt).mean(axis=0)
    order = np.argsort(-mag_true)
    topk_true = order[:k]
    mag_pred = np.abs(dp).mean(axis=0)
    topk_pred = np.argsort(-mag_pred)[:k]
    hit = len(set(topk_true) & set(topk_pred))
    # GO 富集
    top_prots = [prot4422[i] for i in topk_true]
    bg = prot4422
    enrich = fisher_enrich(top_prots, bg, go_of)
    print(f"\n化合物 {c} (n={len(idx)}, max_sim={sim_med[c]:.3f})")
    print(f"  FC PCC = {fc:.4f}")
    print(f"  top-{k} 高效应蛋白命中率 = {hit}/{k} ({hit/k*100:.0f}%)")
    if enrich:
        print(f"  top-{k} 蛋白 GO 富集（top {len(enrich)} 通路）:")
        for g, cnt, f_, sf in enrich:
            print(f"    {g} (命中 {cnt}, 富集 {f_}x, 占比 {sf:.1%})")
    else:
        print("  top-50 蛋白无显著通路富集（阈值 fc>=2, cnt>=2）")
