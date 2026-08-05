# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 06 生成 test 提交（v2.1 定稿模型）
- 只用 test metadata（样本 ID + 条件 + 测量上下文），不加载 test 蛋白组真值文件
- 输出 prediction.csv：4,454 行 × sample_ID + 4,422 蛋白列，log2 尺度，无 NA/inf
"""
import numpy as np, pandas as pd, pickle, hashlib, importlib.util, torch

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
BASE = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- 读训练侧构造（供特征映射复用）----------
meta = pd.read_pickle(f"{DATA}/meta.pkl")
feats = pickle.load(open(f"{DATA}/feats.pkl", 'rb'))
with open(f"{DATA}/prot_names.txt", encoding='utf-8') as f:
    prot_names = f.read().splitlines()
P = len(prot_names)

# ---------- 重建训练侧类别映射（与 02 一致）----------
strains_tr = sorted(meta['Strains'].unique())
chems_tr = sorted(meta.loc[meta['role'].eq('treatment'), 'perturbation_no_concentration'].unique())
strain2id = {s: i for i, s in enumerate(strains_tr)}
chem2id = {c: i for i, c in enumerate(chems_tr)}
med2id = {m: i for i, m in enumerate(sorted(meta['Medium'].unique()))}
tmp2id = {x: i for i, x in enumerate(sorted(meta['Temperature'].unique()))}
sm_cats = sorted((meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str)).unique())
ct_cats = sorted((meta['perturbation_no_concentration'].astype(str) + '|' + meta['Temperature'].astype(str)).unique())
sm2id = {k: i for i, k in enumerate(sm_cats)}; ct2id = {k: i for i, k in enumerate(ct_cats)}
src_cats = sorted(meta['data_source'].unique()); ins_cats = sorted(meta['instrument'].unique()); plt_cats = sorted(meta['Yeast_cell_plate'].unique())
src2id = {k: i for i, k in enumerate(src_cats)}; ins2id = {k: i for i, k in enumerate(ins_cats)}; plt2id = {k: i for i, k in enumerate(plt_cats)}
train_strains = set(meta.loc[meta['split_final'].eq('train'), 'Strains'])
train_chems = set(meta.loc[meta['split_final'].eq('train') & meta['role'].eq('treatment'), 'perturbation_no_concentration'])

def hash_vec(name, dim=32):
    h = hashlib.sha256(str(name).encode()).hexdigest()
    return np.array([int(h[i*2:i*2+2], 16) / 255.0 for i in range(dim)])

# ---------- 读 test metadata ----------
tmeta = pd.read_csv(f"{BASE}/WAYB_WAYC_metadata_test(1).csv")
tmeta = tmeta.set_index('sample_ID')
print(f"[test] 样本数 {len(tmeta)}")

# ---------- 特征编码 ----------
def map_id(s, m, default=-1):
    return m.get(s, default)

strain_id = np.array([map_id(s, strain2id) for s in tmeta['Strains']], dtype=np.int64)
chem_id = np.array([map_id(c, chem2id) for c in tmeta['perturbation_no_concentration']], dtype=np.int64)
chem_hash = np.array([hash_vec(c) for c in tmeta['perturbation_no_concentration']], dtype=np.float32)
med = np.array([[1.0 if m == med else 0.0 for m in sorted(meta['Medium'].unique())] for med in tmeta['Medium']], dtype=np.float32)
temp = ((tmeta['Temperature'].astype(float) - 30.0) / 7.0).values.astype(np.float32)
t = tmeta['pert_time'].astype(float).values
t_log = np.log2(t / 15.0) / np.log2(240.0 / 15.0)
tfeat = np.stack([t_log, np.sin(2*np.pi*t_log), np.cos(2*np.pi*t_log)], axis=1).astype(np.float32)
sm_id = np.array([map_id(f"{s}|{m}", sm2id) for s, m in zip(tmeta['Strains'], tmeta['Medium'])], dtype=np.int64)
ct_id = np.array([map_id(f"{c}|{t_}", ct2id) for c, t_ in zip(tmeta['perturbation_no_concentration'], tmeta['Temperature'])], dtype=np.int64)
src_id = np.array([map_id(s, src2id) for s in tmeta['data_source']], dtype=np.int64)
ins_id = np.array([map_id(s, ins2id) for s in tmeta['instrument']], dtype=np.int64)
plt_id = np.array([map_id(s, plt2id) for s in tmeta['Yeast_cell_plate']], dtype=np.int64)
chem_seen = np.array([1.0 if c in train_chems else 0.0 for c in tmeta['perturbation_no_concentration']], dtype=np.float32)
strain_seen = np.array([1.0 if s in train_strains else 0.0 for s in tmeta['Strains']], dtype=np.float32)
print(f"[特征] test 独有菌株 {int((strain_seen==0).sum())} 样本 | test 独有化合物 {int((chem_seen==0).sum())} 样本")

# ---------- 模型 ----------
_spec = importlib.util.spec_from_file_location("m04", r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model.py")
_m04 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m04)
model = _m04.VCellModel(feats, P=P).to(DEV)
model.load_state_dict(torch.load(f"{DATA}/model_v21.pt", map_location=DEV, weights_only=True))
model.eval()

x = {
    'bio': [torch.from_numpy(strain_id).to(DEV), torch.from_numpy(chem_id).to(DEV), torch.from_numpy(chem_hash).to(DEV),
            torch.from_numpy(med).to(DEV), torch.from_numpy(temp).to(DEV), torch.from_numpy(tfeat).to(DEV),
            torch.from_numpy(sm_id).to(DEV), torch.from_numpy(ct_id).to(DEV)],
    'ctx': [torch.from_numpy(src_id).to(DEV), torch.from_numpy(ins_id).to(DEV), torch.from_numpy(plt_id).to(DEV)],
    'seen': [torch.from_numpy(chem_seen).to(DEV), torch.from_numpy(strain_seen).to(DEV)],
}
with torch.no_grad():
    pred = model(x).cpu().numpy()
print(f"[预测] {pred.shape} | 值域 [{pred.min():.2f}, {pred.max():.2f}]")

# ---------- 提交文件 ----------
sub = pd.DataFrame(pred, index=tmeta.index, columns=prot_names)
sub.index.name = 'sample_ID'
sub.to_csv(f"{DATA}/prediction.csv")
# ---------- 四项校验 ----------
assert len(sub) == 4454, f"样本数 {len(sub)} != 4454"
assert sub.shape[1] == 4422, f"蛋白列 {sub.shape[1]} != 4422"
assert not sub.isna().any().any(), "存在 NA"
assert np.isfinite(sub.values).all(), "存在 inf"
print("校验通过：4,454 行 / 4,422 蛋白列 / 无 NA / 无 inf")
print("[保存] prediction.csv（log2 尺度，声明 prediction_scale=log2）")
