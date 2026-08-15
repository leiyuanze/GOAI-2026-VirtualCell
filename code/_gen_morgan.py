# -*- coding: utf-8 -*-
"""
生成化合物 Morgan 指纹（2048 位，radius=2），替换 chem_hash
输出 data/chem_morgan.pkl: {perturbation_name: fingerprint}
"""
import json
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"

# 1. 读 SMILES
smiles_map = json.load(open(f"{DATA}/chem_smiles.json"))
# 对照与质控
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

# 2. 生成所有 perturbation 的指纹
meta = pd.read_pickle(f"{DATA}/meta.pkl")
all_pert = meta['perturbation_no_concentration'].unique()
print(f"所有 perturbation 值: {len(all_pert)} 个")

fp_map = {}
for p in all_pert:
    if p in smiles_map:
        fp_map[p] = morgan(smiles_map[p]['smiles'])
    else:
        fp_map[p] = np.zeros(2048, dtype=np.float32)  # Quality Control 等
        print(f"  [零向量] {p}")

# 3. 保存
with open(f"{DATA}/chem_morgan.pkl", 'wb') as f:
    pickle.dump(fp_map, f)

# 验证
n_nonzero = sum(1 for v in fp_map.values() if v.sum() > 0)
print(f"指纹图: {len(fp_map)} 个, 非零 {n_nonzero} 个")
print(f"示例 DMSO 指纹前10位: {fp_map.get('DMSO', np.zeros(8))[:10].astype(int)}")
print("Morgan 指纹生成完成")
