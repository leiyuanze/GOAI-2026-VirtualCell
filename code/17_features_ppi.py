# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 17 STRING PPI 蛋白互作边（教程 5.4.4 蛋白互作）
构建高置信度 PPI 边列表，用于图正则（物理互作蛋白协同响应）。
输出：feats['ppi_edges']（E x 2，蛋白索引对）
外部数据来源：STRING v12（4932 酿酒酵母）
"""
import gzip, pickle
import numpy as np

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"

prot_names = [l.strip() for l in open(f"{DATA}/prot_names.txt")]
name2idx = {p: i for i, p in enumerate(prot_names)}
# 特殊名修正
name2idx['OCT1'] = name2idx.get('1-Oct', -1)

# 0. 构建 系统名(oln) -> 标准名(primary) 映射（来自 UniProt）
import csv
oln2prim = {}
with open(f"{DATA}/uniprot_yeast_full.tsv") as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        if len(row) < 5:
            continue
        prim, oln = row[1], row[3]
        for tok in prim.replace(';', ' ').split():
            if tok:
                for o in oln.replace(';', ' ').split():
                    if o:
                        oln2prim.setdefault(o, tok)
print(f"系统名->标准名映射: {len(oln2prim)} 条")

# 1. 解析 links，构建基因名对 + score
edges = []
with gzip.open(f"{DATA}/string_links.txt.gz") as f:
    next(f)  # header
    for line in f:
        parts = line.decode().strip().split()
        if len(parts) < 3:
            continue
        p1, p2, score = parts[0], parts[1], int(parts[2])
        if score < 700:  # 高置信度
            continue
        g1 = p1.split('.')[-1]
        g2 = p2.split('.')[-1]
        edges.append((g1, g2, score))

print(f"高置信度(>=700) PPI 边: {len(edges)} 条")

# 2. 系统名 -> 标准名 -> 蛋白索引 映射
def to_idx(g):
    # 先直接匹配，再通过系统名映射
    if g in name2idx:
        return name2idx[g]
    if g in oln2prim and oln2prim[g] in name2idx:
        return name2idx[oln2prim[g]]
    return -1

ppi_edges = []
for g1, g2, score in edges:
    i = to_idx(g1)
    j = to_idx(g2)
    if i >= 0 and j >= 0 and i != j:
        ppi_edges.append((i, j))

print(f"映射到 4422 蛋白的边: {len(ppi_edges)} 条")

# 3. 保存（去重）
ppi_edges = list(set((min(i,j), max(i,j)) for i, j in ppi_edges))
ppi_edges = np.array(ppi_edges, dtype=np.int64)
print(f"去重后: {ppi_edges.shape}")

feats = pickle.load(open(f"{DATA}/feats.pkl", 'rb'))
feats['ppi_edges'] = ppi_edges
with open(f"{DATA}/feats.pkl", 'wb') as f:
    pickle.dump(feats, f)
print("17 DONE")
