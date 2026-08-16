# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · RDKit descriptors 特征（gpt2 步骤9 / P1-2）
从 chem_smiles.json 为全量 perturbation 生成 10 项 RDKit 描述符：
MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds,
RingCount, FractionCSP3, HeavyAtomCount, FormalCharge
→ feats['chem_desc']（train_val 样本 N×10）+ feats['test_chem_desc']（test 4454×10）
标准化用 train 样本统计量（train-only 合规）
"""
import json, pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
BASE = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"

meta = pd.read_pickle(f"{DATA}/meta.pkl")
tmeta = pd.read_csv(f"{BASE}/WAYB_WAYC_metadata_test(1).csv").set_index('sample_ID')
smiles_map = json.load(open(f"{DATA}/chem_smiles.json", encoding='utf-8'))
smiles_map['DMSO'] = {'smiles': 'CS(=O)C'}
smiles_map['Water'] = {'smiles': 'O'}

DESC_NAMES = ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
              'NumRotatableBonds', 'RingCount', 'FractionCSP3', 'HeavyAtomCount', 'FormalCharge']

def desc_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(len(DESC_NAMES), dtype=np.float32)
    return np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumRings(mol),
        Descriptors.FractionCSP3(mol), mol.GetNumHeavyAtoms(), Chem.GetFormalCharge(mol),
    ], dtype=np.float32)

all_pert = sorted(set(meta['perturbation_no_concentration'].unique()) | set(tmeta['perturbation_no_concentration'].unique()))
pert2desc = {}
missing = []
for p in all_pert:
    if p in smiles_map:
        pert2desc[p] = desc_of(smiles_map[p]['smiles'])
    else:
        pert2desc[p] = np.zeros(len(DESC_NAMES), dtype=np.float32)
        missing.append(p)
print(f"无 SMILES 用零向量: {missing if missing else '无'}")
n_nonzero = sum(1 for v in pert2desc.values() if v.sum() != 0)
print(f"描述符图: {len(pert2desc)} 个, 非零 {n_nonzero} 个")

chem_desc = np.stack([pert2desc[p] for p in meta['perturbation_no_concentration'].values])
test_desc = np.stack([pert2desc[p] for p in tmeta['perturbation_no_concentration'].values])
print(f"chem_desc: {chem_desc.shape} | test_desc: {test_desc.shape}")

# 标准化（train-only）
train_mask = meta['split_final'].eq('train').values
mu = chem_desc[train_mask].mean(axis=0)
sd = chem_desc[train_mask].std(axis=0)
sd[sd == 0] = 1.0
chem_desc = ((chem_desc - mu) / sd).astype(np.float32)
test_desc = ((test_desc - mu) / sd).astype(np.float32)
print(f"标准化后 chem_desc 均值 {chem_desc.mean():.3f} std {chem_desc.std():.3f}")

feats = pickle.load(open(f"{DATA}/feats.pkl", 'rb'))
feats['chem_desc'] = chem_desc
feats['test_chem_desc'] = test_desc
feats['desc_names'] = DESC_NAMES
with open(f"{DATA}/feats.pkl", 'wb') as f:
    pickle.dump(feats, f)
print("feats.pkl 已更新: chem_desc + test_chem_desc")
print("DESC DONE")
