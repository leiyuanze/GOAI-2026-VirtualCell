# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 10 蛋白序列获取（ESM2 输入）
从 UniProt 下载酿酒酵母参考蛋白质组，按基因名映射到 4422 个蛋白。
输出 data/prot_seqs.pkl: {prot_name: aa_sequence}
外部数据来源：UniProt（organism_id:559292, reviewed），版本 2026-08
"""
import csv
import pickle
import urllib.request

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"

# 1. 下载（如已有则跳过）
import os
tsv_path = f"{DATA}/uniprot_yeast_full.tsv"
if not os.path.exists(tsv_path):
    url = ('https://rest.uniprot.org/uniprotkb/stream'
           '?query=organism_id:559292+AND+reviewed:true'
           '&format=tsv&fields=accession,gene_primary,gene_synonym,gene_oln,sequence')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    data = urllib.request.urlopen(req, timeout=180).read().decode('utf-8')
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write(data)
    print(f"下载 UniProt 参考蛋白组 -> {tsv_path}")

# 2. 构建 名称->序列 映射（基因名字段按 ; 和空格拆分）
name2seq = {}
with open(tsv_path) as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        if len(row) < 5:
            continue
        acc, prim, syn, oln, seq = row[0], row[1], row[2], row[3], row[4]
        for field in (prim, syn, oln):
            for tok in field.replace(';', ' ').split():
                tok = tok.strip()
                if tok:
                    name2seq.setdefault(tok, seq)
        name2seq.setdefault(acc, seq)

# 3. 映射 prot_names
prot_names = [l.strip() for l in open(f"{DATA}/prot_names.txt")]
# 特殊名修正：'1-Oct' 是 Excel 把 OCT1 污染成日期格式
SPECIAL = {'1-Oct': 'OCT1'}

prot_seqs = {}
miss = []
for p in prot_names:
    q = SPECIAL.get(p, p)
    if q in name2seq:
        prot_seqs[p] = name2seq[q]
    else:
        miss.append(p)

print(f"映射: {len(prot_seqs)}/{len(prot_names)} ({len(prot_seqs)/len(prot_names)*100:.1f}%)")
if miss:
    print(f"未命中 {len(miss)}: {miss}")

# 4. 保存
with open(f"{DATA}/prot_seqs.pkl", 'wb') as f:
    pickle.dump(prot_seqs, f)

# 统计序列长度
lens = [len(s) for s in prot_seqs.values()]
print(f"序列长度: min={min(lens)}, max={max(lens)}, mean={sum(lens)/len(lens):.0f}")
print("10 DONE")
