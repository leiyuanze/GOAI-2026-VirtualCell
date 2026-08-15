# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 16 完整 ESM2（320维，不压缩）+ PPI 蛋白互作先验
1. esm2_emb 改为完整 320 维（归一化）
2. 下载 STRING PPI，构建蛋白互作图，用于图正则
外部数据来源：ESM2(Meta), STRING v12
"""
import pickle
import numpy as np

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
feats = pickle.load(open(f"{DATA}/feats.pkl", 'rb'))

# 1. 完整 ESM2（320维）
esm2 = np.load(f"{DATA}/prot_esm2.npy").astype(np.float32)  # 4422 x 320
esm2 = esm2 / (np.linalg.norm(esm2, axis=1, keepdims=True) + 1e-6)
feats['esm2_emb'] = esm2
feats['esm_dim'] = 320
print(f"[1] esm2_emb 完整 {esm2.shape}")

with open(f"{DATA}/feats.pkl", 'wb') as f:
    pickle.dump(feats, f)
print("16 DONE")
