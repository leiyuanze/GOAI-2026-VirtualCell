# -*- coding: utf-8 -*-
import sys
sys.stdout = open("bench_result.txt", "w")
"""GPU 基准测试：测前向速度瓶颈"""
import numpy as np, pandas as pd, pickle, torch, time, importlib.util

DATA = 'data'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
feats = pickle.load(open(f'{DATA}/feats.pkl','rb'))
P = 4422

_s = importlib.util.spec_from_file_location('m30', '04_model_v30.py')
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
model = _m.VCellModel(feats, P=P).to(DEV).eval()
print(f"设备 {DEV}, 显存 {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print(f"模型参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

ctx_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)
morgan_all = feats['chem_morgan']
N = len(meta)
B = 128

def make_x(idx):
    return {
        'bio': [torch.from_numpy(feats['strain_id'][idx]), torch.from_numpy(feats['chem_id'][idx]),
                torch.from_numpy(feats['chem_hash'][idx]), torch.from_numpy(feats['medium_onehot'][idx]),
                torch.from_numpy(feats['temp_norm'][idx]), torch.from_numpy(feats['time_feat'][idx]),
                torch.from_numpy(feats['sm_id'][idx]), torch.from_numpy(feats['ct_id'][idx])],
        'ctx': [torch.from_numpy(feats['src_id'][idx]), torch.from_numpy(feats['ins_id'][idx]),
                torch.from_numpy(feats['plt_id'][idx])],
        'seen': [torch.from_numpy(feats['chem_seen'][idx]), torch.from_numpy(feats['strain_seen'][idx])],
        'ctx_prior': torch.from_numpy(ctx_all[idx]),
        'chem_morgan': torch.from_numpy(morgan_all[idx]),
    }

def to_gpu(x):
    return {k: (v.to(DEV) if k in ('ctx_prior','chem_morgan') else [t.to(DEV) for t in v]) for k, v in x.items()}

# 方式1：当前方式（每 batch make_x + to GPU）
idx = np.arange(2000)
t0 = time.time()
with torch.no_grad():
    for i in range(0, 2000, B):
        x = make_x(idx[i:i+B])
        model(to_gpu(x))
t1 = time.time()
n_batch = 2000 // B
print(f"方式1(每batch构造+传输): {n_batch} batch 用时 {t1-t0:.2f}s = {(t1-t0)/n_batch*1000:.1f}ms/batch")

# 方式2：预加载到 GPU，只索引
strain_id_g = torch.from_numpy(feats['strain_id']).to(DEV)
chem_id_g = torch.from_numpy(feats['chem_id']).to(DEV)
chem_hash_g = torch.from_numpy(feats['chem_hash']).to(DEV)
medium_g = torch.from_numpy(feats['medium_onehot']).to(DEV)
temp_g = torch.from_numpy(feats['temp_norm']).to(DEV)
time_g = torch.from_numpy(feats['time_feat']).to(DEV)
sm_g = torch.from_numpy(feats['sm_id']).to(DEV)
ct_g = torch.from_numpy(feats['ct_id']).to(DEV)
src_g = torch.from_numpy(feats['src_id']).to(DEV)
ins_g = torch.from_numpy(feats['ins_id']).to(DEV)
plt_g = torch.from_numpy(feats['plt_id']).to(DEV)
cseen_g = torch.from_numpy(feats['chem_seen']).to(DEV)
sseen_g = torch.from_numpy(feats['strain_seen']).to(DEV)
ctx_g = torch.from_numpy(ctx_all).to(DEV)
morgan_g = torch.from_numpy(morgan_all).to(DEV)
print(f"预加载特征到 GPU 后显存占用: {torch.cuda.memory_allocated()/1e9:.2f}GB")

t0 = time.time()
with torch.no_grad():
    for i in range(0, 2000, B):
        b = idx[i:i+B]
        x = {'bio':[strain_id_g[b],chem_id_g[b],chem_hash_g[b],medium_g[b],temp_g[b],time_g[b],sm_g[b],ct_g[b]],
             'ctx':[src_g[b],ins_g[b],plt_g[b]],
             'seen':[cseen_g[b],sseen_g[b]],
             'ctx_prior':ctx_g[b],'chem_morgan':morgan_g[b]}
        model(x)
t1 = time.time()
print(f"方式2(预加载GPU,只索引): {n_batch} batch 用时 {t1-t0:.2f}s = {(t1-t0)/n_batch*1000:.1f}ms/batch")
print(f"加速比: {(2000//B)} 方式1/方式2 = {( (2000//B) ) }")
