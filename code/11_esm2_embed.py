# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 11 ESM2 蛋白 embedding 提取
用 facebook/esm2_t6_8M_UR50D (320维) 对 4422 个蛋白序列提取 mean-pooled 表示。
输出 data/prot_esm2.npy (4422 x 320)
外部数据来源：ESM2 (Meta, UniProt 训练)，模型 esm2_t6_8M_UR50D
"""
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MAX_LEN = 1024  # t6_8M 的上下文长度

# 1. 读序列（按 prot_names 顺序）
prot_names = [l.strip() for l in open(f"{DATA}/prot_names.txt")]
prot_seqs = pickle.load(open(f"{DATA}/prot_seqs.pkl", 'rb'))
seqs = [prot_seqs[p] for p in prot_names]
print(f"[1] {len(seqs)} 个蛋白序列")

# 2. 加载模型
print(f"[2] 加载 {MODEL_NAME} ...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
print(f"    参数 {sum(p.numel() for p in model.parameters())/1e6:.1f}M, 设备 {device}")

# 3. 批量提取
embeddings = []
BATCH = 32
with torch.no_grad():
    for i in range(0, len(seqs), BATCH):
        batch_seqs = seqs[i:i+BATCH]
        inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True,
                           truncation=True, max_length=MAX_LEN)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        # mean pooling（忽略 padding）
        mask = inputs['attention_mask'].unsqueeze(-1)
        emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings.append(emb.cpu().numpy())
        if (i // BATCH) % 20 == 0:
            print(f"    进度 {i}/{len(seqs)}")

emb = np.vstack(embeddings).astype(np.float32)
print(f"[3] embedding 形状: {emb.shape}")

# 4. 保存
np.save(f"{DATA}/prot_esm2.npy", emb)
print(f"[4] 已保存 {DATA}/prot_esm2.npy")
print("11 DONE")
