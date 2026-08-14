# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 15 GO 通路分组（教程 5.4.4 通路归属）
从 UniProt GO 生物过程注释，构建「高频 GO term -> 蛋白」binary 矩阵。
用于模型通路注意力：同一通路蛋白共享权重偏置。
输出：feats['go_mat']（K x P，K 个高频 GO term），feats['go_names']
外部数据来源：UniProt GO 注释（organism_id:559292, reviewed）
"""
import re, pickle
import numpy as np
from collections import Counter

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"

prot_names = [l.strip() for l in open(f"{DATA}/prot_names.txt")]
P = len(prot_names)

# 1. 读 GO 注释，构建 gene -> GO terms
gene2go = {}
with open(f"{DATA}/uniprot_go.tsv") as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 4:
            continue
        gene_prim, gene_oln, go_p = parts[1], parts[2], parts[3]
        go_ids = re.findall(r'GO:\d+', go_p)
        if not go_ids:
            continue
        # 用 gene name 和 ordered locus name 都映射
        for gene in [gene_prim, gene_oln]:
            if gene:
                gene2go.setdefault(gene, set()).update(go_ids)

# 2. 映射到 4422 蛋白
prot2go = {}
for p in prot_names:
    q = 'OCT1' if p == '1-Oct' else p  # Excel 污染
    if q in gene2go:
        prot2go[p] = gene2go[q]

print(f"有 GO 注释的蛋白: {len(prot2go)}/{P}")

# 3. 统计 GO term 频率
go_counter = Counter()
for go_set in prot2go.values():
    go_counter.update(go_set)

# 筛选高频 GO term（出现在 >= 30 个蛋白里）
min_count = 30
high_go = [(go, c) for go, c in go_counter.items() if c >= min_count]
high_go.sort(key=lambda x: -x[1])
print(f"高频 GO term (>= {min_count} 蛋白): {len(high_go)} 个")

# 4. 构建 binary 矩阵 K x P
K = min(len(high_go), 200)  # 最多 200 个 GO term
go_list = [go for go, _ in high_go[:K]]
go2idx = {go: i for i, go in enumerate(go_list)}
go_mat = np.zeros((K, P), dtype=np.float32)
for j, p in enumerate(prot_names):
    if p in prot2go:
        for go in prot2go[p]:
            if go in go2idx:
                go_mat[go2idx[go], j] = 1.0

coverage = (go_mat.sum(axis=0) > 0).mean()
print(f"GO 矩阵: {go_mat.shape}，覆盖 {coverage*100:.1f}% 蛋白（至少属于 1 个高频通路）")
print(f"平均每个蛋白属于 {(go_mat.sum(axis=0)[go_mat.sum(axis=0)>0]).mean():.1f} 个高频通路")
print(f"前 10 个高频通路（蛋白数）: {[(go, c) for go, c in high_go[:10]]}")

# 5. 保存
feats = pickle.load(open(f"{DATA}/feats.pkl", 'rb'))
feats['go_mat'] = go_mat
feats['go_names'] = go_list
feats['n_go'] = K
with open(f"{DATA}/feats.pkl", 'wb') as f:
    pickle.dump(feats, f)
print("15 DONE")
