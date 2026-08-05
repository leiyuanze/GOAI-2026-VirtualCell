# -*- coding: utf-8 -*-
"""通路富集分析：用预测Δ找生物学通路证据"""
import numpy as np, pandas as pd, pickle, torch, importlib.util, urllib.request, gzip, io
from scipy.stats import fisher_exact

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float32)
mask = np.load(f'{DATA}/mask.npy').astype(np.float32)
feats = pickle.load(open(f'{DATA}/feats.pkl','rb'))
P = y_log2.shape[1]
ctx_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)
tr_y_nan = np.where(mask.astype(bool), y_log2, np.nan)
with open(f'{DATA}/prot_names.txt') as f:
    prot_names = f.read().splitlines()
gene2i = {g: i for i, g in enumerate(prot_names)}

# ---------- Download yeast GO annotations ----------
print("[GO] 下载酿酒酵母 GO 注释...")
url = "https://current.geneontology.org/annotations/sgd.gaf.gz"
try:
    resp = urllib.request.urlopen(url, timeout=30)
    with gzip.GzipFile(fileobj=io.BytesIO(resp.read())) as f:
        lines = f.read().decode().split('\n')
except:
    print("[GO] 网络不可用，跳过外部数据。用蛋白质名称做简单功能分组...")
    lines = []

go_map = {}  # gene → set of GO terms
for line in lines:
    if line.startswith('!'): continue
    parts = line.split('\t')
    if len(parts) < 5: continue
    gene, go_term = parts[2], parts[4]
    if gene not in gene2i: continue
    go_map.setdefault(gene, set()).add(go_term)

# GO term → name
go_names = {}
for line in lines:
    if line.startswith('!'): continue
    parts = line.split('\t')
    if len(parts) < 10: continue
    go_names[parts[4]] = parts[9] if parts[9] else parts[4]

print(f"[GO] 覆盖 {len(go_map)} 个蛋白, {len(go_names)} 个 GO term")

# ---------- Load model & matched control ----------
_s21 = importlib.util.spec_from_file_location('m21', r'D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v21.py')
_m21 = importlib.util.module_from_spec(_s21); _s21.loader.exec_module(_m21)
_s29 = importlib.util.spec_from_file_location('m29', r'D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v29.py')
_m29 = importlib.util.module_from_spec(_s29); _s29.loader.exec_module(_m29)

def load(path, cls, sa=False):
    m = cls(feats, P=P); m.load_state_dict(torch.load(f'{DATA}/{path}', map_location=DEV, weights_only=True))
    if sa: m.set_strain_avg()
    return m.to(DEV).eval()

m21 = load('model_v21.pt', _m21.VCellModel)
m21s43 = load('model_v21_s43.pt', _m21.VCellModel)
m21s44 = load('model_v21_s44.pt', _m21.VCellModel)
m29 = load('model_v29_best.pt', _m29.VCellModel, sa=True)
models = [(m21,'v21'),(m21s43,'v21'),(m21s44,'v21'),(m29,'v29')]

ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = (meta.iloc[ctrl_idx]['data_source'].astype(str)+'|'+meta.iloc[ctrl_idx]['instrument'].astype(str)+'|'
            +meta.iloc[ctrl_idx]['Yeast_cell_plate'].astype(str)+'|'+meta.iloc[ctrl_idx]['Strains'].astype(str)+'|'
            +meta.iloc[ctrl_idx]['Medium'].astype(str)+'|'+meta.iloc[ctrl_idx]['Temperature'].astype(str)+'|'
            +meta.iloc[ctrl_idx]['pert_time'].astype(str)).values
ctrl_lookup={}
for k,p in zip(ctrl_key,ctrl_idx): ctrl_lookup.setdefault(k,[]).append(p)

def mc_mean(sid):
    r=meta.iloc[sid]
    k=(str(r['data_source'])+'|'+str(r['instrument'])+'|'+str(r['Yeast_cell_plate'])+'|'
       +str(r['Strains'])+'|'+str(r['Medium'])+'|'+str(r['Temperature'])+'|'+str(r['pert_time']))
    if k not in ctrl_lookup: return np.full(P, np.nan)
    rows=ctrl_lookup[k]; cv=tr_y_nan[rows]; cm=mask[rows]>0
    with np.errstate(invalid='ignore'):
        return np.where(cm.sum(0)>0,np.nansum(np.where(cm,cv,np.nan),0)/cm.sum(0),np.nan)

def predict(idx):
    preds=[]
    for m,tag in models:
        x={'bio':[torch.from_numpy(feats['strain_id'][idx]),torch.from_numpy(feats['chem_id'][idx]),
                  torch.from_numpy(feats['chem_hash'][idx]),torch.from_numpy(feats['medium_onehot'][idx]),
                  torch.from_numpy(feats['temp_norm'][idx]),torch.from_numpy(feats['time_feat'][idx]),
                  torch.from_numpy(feats['sm_id'][idx]),torch.from_numpy(feats['ct_id'][idx])],
           'ctx':[torch.from_numpy(feats['src_id'][idx]),torch.from_numpy(feats['ins_id'][idx]),torch.from_numpy(feats['plt_id'][idx])],
           'seen':[torch.from_numpy(feats['chem_seen'][idx]),torch.from_numpy(feats['strain_seen'][idx])]}
        if tag=='v29': x['ctx_prior']=torch.from_numpy(ctx_all[idx])
        with torch.no_grad():
            xg={k:(v.to(DEV) if k=='ctx_prior' else [t.to(DEV) for t in v]) for k,v in x.items()}
            preds.append(m(xg).cpu().numpy())
    return np.mean(preds,axis=0)

# ---------- 计算预测 Δ 并做富集 ----------
print("\n" + "=" * 70)
print("通路富集分析（Fisher Exact Test, top 200 |Δ| 蛋白 vs 背景）")
print("=" * 70)

for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
    idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    if len(idx) == 0: continue
    
    pred = predict(idx)
    yc = np.array([mc_mean(s) for s in idx])
    dp = pred - yc  # predicted Δ
    
    # 取所有样本的 |Δ| 均值最大的 top 200 蛋白
    abs_dp = np.nanmean(np.abs(dp), axis=0)
    top200 = np.argsort(abs_dp)[-200:]
    top_genes = set(prot_names[i] for i in top200 if prot_names[i] in go_map)
    
    background_genes = set(g for g in prot_names if g in go_map)
    
    if not top_genes: continue
    
    # Fisher exact per GO term
    results = []
    go_terms = set()
    for g in top_genes: go_terms.update(go_map.get(g, set()))
    
    for go in go_terms:
        go_genes = set(g for g in background_genes if go in go_map.get(g, set()))
        if len(go_genes) < 5 or len(go_genes) > 500: continue
        
        a = len(top_genes & go_genes)  # top200 & in pathway
        b = len(top_genes - go_genes)   # top200 & not in pathway
        c = len(go_genes - top_genes)   # not top200 & in pathway
        d = len(background_genes - top_genes - go_genes)  # not top200 & not in pathway
        
        if a < 2: continue
        _, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
        results.append((go, go_names.get(go, go), a, float(pval)))
    
    results.sort(key=lambda x: x[3])
    
    print(f"\n--- {scene} (样本数={len(idx)}) ---")
    print(f"  Top200 |Δ| 蛋白中 {len(top_genes)} 个有 GO 注释")
    print(f"  {'GO Term':<12} {'P-value':<8} {'Hits':>5}  {'描述'}")
    for go, name, hits, pval in results[:10]:
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {go:<12} {pval:<8.2e} {hits:>5}  {name[:60]} {sig}")

# ---------- 高效应蛋白列表 ----------
print()
print("=" * 70)
print("关键高效应蛋白（|Δ| > 1，全局排名）")
print("=" * 70)

all_abs = np.nanmean(np.abs(dp), axis=0)
top_hi = np.where(all_abs > 1.0)[0]
print(f"  |Δ| > 1 的蛋白数: {len(top_hi)}")
print(f"  {'蛋白名':<10} {'|Δ|均值':>8}  {'GO功能（前3个）'}")
for pi in top_hi[:20]:
    g = prot_names[pi]
    go_funcs = []
    for go in list(go_map.get(g, set()))[:3]:
        go_funcs.append(go_names.get(go, go)[:40])
    print(f"  {g:<10} {all_abs[pi]:>8.3f}  {'; '.join(go_funcs)}")

print("\nDONE")
