# -*- coding: utf-8 -*-
"""
解析 1011 项目 SNP 距离矩阵，提取 6 个菌株（BAH/BAI/CEK/CGD/CRD/DHY210）的遗传距离。
输出菌株相似度，用于 unseen 菌株（CRD）的响应迁移。
"""
import gzip, numpy as np

strains = ['BAH', 'BAI', 'CEK', 'CGD', 'CRD', 'DHY210']

# 读距离矩阵（tab 分隔，首行首列是菌株名）
with gzip.open('data/1011_SNP_distance.tab.gz', 'rb') as f:
    raw = f.read().decode('utf-8')

lines = raw.strip().split('\n')
header = lines[0].strip().split('\t')
print(f'矩阵维度: {len(lines)-1} 行 x {len(header)} 列')

# 找 6 个菌株在矩阵里的位置
col_names = header[1:]  # 第一列可能是空或菌株名
print(f'列名前10: {col_names[:10]}')

# 检查菌株是否在矩阵里
found = {}
for s in strains:
    if s in col_names:
        found[s] = col_names.index(s)
    else:
        print(f'[缺失] {s} 不在列名里')

print(f'找到 {len(found)}/6 个菌株: {list(found.keys())}')

# 提取距离子矩阵
if len(found) == 6:
    idx = [found[s] for s in strains]
    D = np.zeros((6, 6))
    row_strains = []
    for line in lines[1:]:
        parts = line.strip().split('\t')
        row_name = parts[0]
        if row_name in strains:
            vals = [float(parts[i+1]) for i in idx]  # 对应 6 个菌株列
            row_strains.append(row_name)
            if len(D) == 6 and len(row_strains) <= 6:
                D[strains.index(row_name)] = vals
    print('\n遗传距离矩阵（6 菌株）:')
    print(f'{"":>10}' + ''.join(f'{s:>10}' for s in strains))
    for i, s in enumerate(strains):
        print(f'{s:>10}' + ''.join(f'{D[i,j]:>10.4f}' for j in range(6)))
    np.save('data/strain_distance.npy', D)
    np.save('data/strain_order.npy', np.array(strains))
    print('\n已保存 strain_distance.npy + strain_order.npy')
