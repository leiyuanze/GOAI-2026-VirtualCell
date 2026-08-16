# -*- coding: utf-8 -*-
"""
gpt2 步骤9：Morgan(2048) + RDKit desc(10) 拼接 → PCA(32, whiten)，train-only fit
→ feats['chem_fp32']（train_val N×32）+ feats['test_chem_fp32']（test 4454×32）
"""
import json, pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.decomposition import PCA

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
BASE = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"

meta = pd.read_pickle(f"{DATA}/meta.pkl")
tmeta = pd.read_csv(f"{BASE}/WAYB_WAYC_metadata_test(1).csv").set_index('sample_ID')
smiles_map = json.load(open(f"{DATA}/chem_smiles.json", encoding='utf-8'))
smiles_map['DMSO'] = {'smiles': 'CS(=O)C'}
smiles_map['Water'] = {'smiles': 'O'}

DESC_NAMES = ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
              'NumRotatableBonds', 'RingCount', 'FractionCSP3', 'HeavyAtomCount', 'FormalCharge']

def fp_desc_of(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(2048, dtype=np.float32), np.zeros(len(DESC_NAMES), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_arr = np.frombuffer(fp.ToBitString().encode(), dtype='u1').astype(np.float32) - 48
    desc = np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumRings(mol),
        Descriptors.FractionCSP3(mol), mol.GetNumHeavyAtoms(), Chem.GetFormalCharge(mol),
    ], dtype=np.float32)
    return fp_arr, desc

all_pert = sorted(set(meta['perturbation_no_concentration'].unique()) | set(tmeta['perturbation_no_concentration'].unique()))
pert_fp = {}; pert_desc = {}
for p in all_pert:
    if p in smiles_map:
        pert_fp[p], pert_desc[p] = fp_desc_of(smiles_map[p]['smiles'])
    else:
        pert_fp[p] = np.zeros(2048, dtype=np.float32)
        pert_desc[p] = np.zeros(len(DESC_NAMES), dtype=np.float32)

fp_all = np.stack([pert_fp[p] for p in meta['perturbation_no_concentration'].values])
desc_all = np.stack([pert_desc[p] for p in meta['perturbation_no_concentration'].values])
fp_test = np.stack([pert_fp[p] for p in tmeta['perturbation_no_concentration'].values])
desc_test = np.stack([pert_desc[p] for p in tmeta['perturbation_no_concentration'].values])
print(f"fp_all {fp_all.shape} desc_all {desc_all.shape} | fp_test {fp_test.shape}")

# 拼接 + 标准化（train-only）
train_mask = meta['split_final'].eq('train').values
X = np.concatenate([fp_all, desc_all], axis=1)
Xt = np.concatenate([fp_test, desc_test], axis=1)
mu = X[train_mask].mean(axis=0); sd = X[train_mask].std(axis=0) + 1e-8
X = ((X - mu) / sd).astype(np.float32)
Xt = ((Xt - mu) / sd).astype(np.float32)

# PCA(32, whiten)，只用 train 化合物拟合（gpt2 步骤9）
pca = PCA(n_components=32, whiten=True, random_state=42)
pca.fit(X[train_mask])
chem_fp32 = pca.transform(X).astype(np.float32)
test_chem_fp32 = pca.transform(Xt).astype(np.float32)
print(f"chem_fp32 {chem_fp32.shape} | test_chem_fp32 {test_chem_fp32.shape} | 方差 {pca.explained_variance_ratio_.sum()*100:.1f}%")

feats = pickle.load(open(f"{DATA}/feats.pkl", 'rb'))
feats['chem_fp32'] = chem_fp32
feats['test_chem_fp32'] = test_chem_fp32
pickle.dump(feats, open(f"{DATA}/feats.pkl", 'wb'))
print("feats 已更新: chem_fp32 / test_chem_fp32 写入")
