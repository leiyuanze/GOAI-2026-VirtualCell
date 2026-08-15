# -*- coding: utf-8 -*-
"""
重大空白修复：test 新化合物的 Morgan 指纹
问题：feats['chem_morgan'] 是按 train_val 样本做的 PCA（SVD），
      test 新化合物（Camptothecin/G418/MMS 等 11 个）从未有指纹，
      推理时映射查不到 → 零向量 → C 分支对它们无结构信息。
修复：
1. 用下载好的 SMILES + RDKit 生成 2048 指纹
2. 用 feats 现有的 PCA 变换（U[:, :64]*S[:64] 是从 train_val 指纹学到的）
   把新指纹投影到同一 64 维空间（与训练一致，合规）
3. 更新 chem_morgan.pkl（全量指纹图）+ 生成 test 行索引的 test_chem_morgan64
"""
import json, pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

DATA = 'data'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
tmeta = pd.read_csv('../input/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))

# 1. 读 SMILES（已含新下载的 11 个）
smiles_map = json.load(open(f'{DATA}/chem_smiles.json', encoding='utf-8'))
smiles_map['DMSO'] = {'smiles': 'CS(=O)C'}
smiles_map['Water'] = {'smiles': 'O'}

def morgan(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# 2. 重建全量指纹图（train_val 43 + test 新 11 + 对照/质控）
#    train_val 的指纹从现有 feats 反推会丢信息，直接重新生成（用 chem_smiles.json 全量）
all_pert = sorted(set(meta['perturbation_no_concentration'].unique()) | set(tmeta['perturbation_no_concentration'].unique()))
print(f'全量 perturbation 值: {len(all_pert)} 个')

fp_map = {}
missing = []
for p in all_pert:
    if p in smiles_map:
        fp_map[p] = morgan(smiles_map[p]['smiles'])
    else:
        fp_map[p] = np.zeros(2048, dtype=np.float32)
        missing.append(p)
print(f'无 SMILES 用零向量: {missing if missing else "无"}')
n_nonzero = sum(1 for v in fp_map.values() if v.sum() > 0)
print(f'指纹图: {len(fp_map)} 个, 非零 {n_nonzero} 个')

# 3. 保存全量指纹图
with open(f'{DATA}/chem_morgan.pkl', 'wb') as f:
    pickle.dump(fp_map, f)
print('已更新 chem_morgan.pkl')

# 4. 用 train_val 指纹重建 PCA 变换（与 12_features_v30.py 一致），
#    然后投影所有化合物（含 test 新）到同一 64 维空间
train_fps = np.stack([fp_map[p] for p in meta['perturbation_no_concentration'].values])  # (N, 2048)
morgan_c = train_fps - train_fps.mean(axis=0)
# SVD: morgan_c (N,2048) = U (N,N) @ diag(S) @ Vt (2048,2048)
U, S, Vt = np.linalg.svd(morgan_c, full_matrices=False)
# 样本空间投影: X @ Vt.T -> 每个样本在 64 个主成分上的坐标
# 新指纹投影: (fp - mean) @ Vt[:64].T  (Vt 是 (min(N,2048), 2048) = (2048, 2048))
def project(fp):
    c = fp - train_fps.mean(axis=0)
    v = c @ Vt[:64].T  # (64,)
    v = v / (np.linalg.norm(v) + 1e-6)
    return v.astype(np.float32)

# 5. 更新 feats['chem_morgan']（train_val 行，重新投影保证与 64 维空间一致）
feats['chem_morgan'] = np.stack([project(fp_map[p]) for p in meta['perturbation_no_concentration'].values])
print(f"feats['chem_morgan'] 更新: {feats['chem_morgan'].shape}")

# 6. 新增 test 行索引的指纹（供推理脚本用）
pert2morgan64 = {p: project(fp_map[p]) for p in all_pert}
test_morgan64 = np.stack([pert2morgan64[p] for p in tmeta['perturbation_no_concentration'].values])
feats['test_chem_morgan'] = test_morgan64
feats['pert2morgan64'] = pert2morgan64
print(f"feats['test_chem_morgan']: {test_morgan64.shape}")

# 7. 验证 test 新化合物非零
print('\ntest 新化合物指纹状态:')
idx = np.where(tmeta['split_final'].eq('test_chem_only').values)[0]
t_chems = sorted(tmeta.iloc[idx]['perturbation_no_concentration'].unique())
for c in t_chems:
    v = pert2morgan64.get(c, np.zeros(64))
    print(f'  {c}: norm={np.linalg.norm(v):.3f}')

with open(f'{DATA}/feats.pkl', 'wb') as f:
    pickle.dump(feats, f)
print('\nfeats.pkl 已保存')
