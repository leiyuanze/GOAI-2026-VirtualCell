# -*- coding: utf-8 -*-
"""验证基因拷贝数相似度对 unseen 菌株 CRD 的迁移收益（对比 SNP 距离）"""
import gzip, numpy as np, pandas as pd, pickle

# 读拷贝数矩阵
raw = gzip.open('data/genesMatrix_CopyNumber.tab.gz','rb').read().decode('utf-8')
lines = raw.strip().split('\n')
header = lines[0].strip().split('\t')
gene_cols = header[1:]  # 基因列
print(f'拷贝数矩阵: {len(lines)-1} 菌株 x {len(gene_cols)} 基因')

strains5 = ['BAH','BAI','CEK','CGD','CRD']
cnv = {}
for line in lines[1:]:
    parts = line.strip().split('\t')
    s = parts[0]
    if s in strains5:
        cnv[s] = np.array([float('nan') if x == 'NA' else float(x) for x in parts[1:]], dtype=np.float32)
print(f'找到 {len(cnv)}/5 菌株的拷贝数: {list(cnv.keys())}')

# 计算菌株间拷贝数相似度（相关性，nan-aware）
def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10: return 0.0
    a = a[ok] - a[ok].mean(); b = b[ok] - b[ok].mean()
    return float((a*b).sum() / np.sqrt((a*a).sum()*(b*b).sum())) if (a*a).sum()>0 and (b*b).sum()>0 else 0.0

print('\n菌株间拷贝数相似度（相关性）:')
print('        ' + ''.join(f'{s:>10}' for s in strains5))
for s in strains5:
    row = ''.join(f'{corr(cnv[s], cnv[t]):>10.3f}' for t in strains5)
    print(f'{s:>8}{row}')

# 验证迁移收益：CRD 用拷贝数相似度加权训练菌株响应
DATA = 'data'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
y = np.load(f'{DATA}/y_log2.npy'); mask = np.load(f'{DATA}/mask.npy').astype(bool)
feats = pickle.load(open(f'{DATA}/feats.pkl','rb'))
gmean = feats['gmean']; strain_means = feats['strain_means']
strains = sorted(meta['Strains'].unique()); s2i = {s:i for i,s in enumerate(strains)}
train_strains = ['BAH','CEK','CGD','DHY210']

# test CRD 真值
tmeta = pd.read_csv('../input/WAYB_WAYC_metadata_test(1).csv')
tprot = pd.read_csv('../input/WAYB_WAYC_proteome_raw_test.csv', index_col='sample_ID')
prot_names = [l.strip() for l in open(f'{DATA}/prot_names.txt')]
tprot = tprot[prot_names]; tprot_log2 = np.log2(tprot.values)
crd_rows = tmeta['Strains'].eq('CRD').values
crd_mean = np.nanmean(np.where(np.isfinite(tprot_log2[crd_rows]), tprot_log2[crd_rows], np.nan), axis=0)
ctrl_rows = meta['split_final'].eq('train').values & meta['role'].eq('control').values
ctrl_mean = np.nanmean(np.where(mask[ctrl_rows], y[ctrl_rows], np.nan), axis=0)
crd_specific = crd_mean - ctrl_mean

# 拷贝数相似度加权（DHY210 不在矩阵，相似度设为 0）
sim_cnv = np.array([corr(cnv['CRD'], cnv[t]) if t in cnv else 0.0 for t in train_strains])
sim_cnv = np.exp(sim_cnv / 0.2)  # 放大（相关性范围小）
sim_cnv = sim_cnv / sim_cnv.sum()
pred_cnv = sum(sim_cnv[j] * (strain_means[s2i[train_strains[j]]] - gmean) for j in range(4))
pred_avg = np.mean(np.stack([strain_means[s2i[t]] - gmean for t in train_strains]), axis=0)

def pcorr(a,b):
    ok = np.isfinite(a)&np.isfinite(b)
    return float(np.corrcoef(a[ok],b[ok])[0,1]) if ok.sum()>100 else float('nan')

print(f'\n拷贝数相似度权重(CRD): ' + ', '.join(f'{t}={sim_cnv[j]:.3f}' for j,t in enumerate(train_strains)))
print(f'拷贝数加权预测 vs CRD真实: {pcorr(pred_cnv, crd_specific):.4f}')
print(f'平均预测 vs CRD真实: {pcorr(pred_avg, crd_specific):.4f}')
print(f'提升: {pcorr(pred_cnv, crd_specific) - pcorr(pred_avg, crd_specific):+.4f}')
