# -*- coding: utf-8 -*-
"""通路富集分析 v2：用酵母基因命名规律做功能分组 + 高效应蛋白分析"""
import numpy as np, pandas as pd, pickle, torch, importlib.util
from scipy.stats import mannwhitneyu

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
gene2i = {g:i for i,g in enumerate(prot_names)}

# ---------- 酵母功能分组（基于基因命名规律）----------
PATHWAYS = {
    '半乳糖代谢': ['GAL1','GAL2','GAL3','GAL4','GAL7','GAL10','GAL11','GAL80','MIG1','HXK1','HXK2','PGM1','PGM2','UGP1'],
    '糖酵解/糖异生': ['HXK1','HXK2','GLK1','PGI1','PFK1','PFK2','FBA1','TPI1','TDH1','TDH2','TDH3','PGK1','GPM1','ENO1','ENO2','PYK2','CDC19','PDC1','PDC5','PDC6','ADH1','ADH2','ADH3','ADH4','ADH5','FBP1','PCK1','MDH2'],
    '核糖体蛋白(大亚基)': [g for g in prot_names if g.startswith('RPL')],
    '核糖体蛋白(小亚基)': [g for g in prot_names if g.startswith('RPS')],
    '热休克/应激': ['HSP10','HSP12','HSP26','HSP30','HSP42','HSP60','HSP82','HSP104','HSP150','SSA1','SSA2','SSA3','SSA4','SSB1','SSB2','SSC1','SSD1','SSE1','SSE2','SSZ1','STI1','YDJ1','SIS1','ZPR1','HSF1','MSN2','MSN4','HOG1','PBS2','SLT2','MKK1','MKK2'],
    '氨基酸合成': ['ARG1','ARG3','ARG4','ARG5','ARG8','ARG80','ARG81','HIS1','HIS2','HIS3','HIS4','HIS5','HIS6','HIS7','LEU1','LEU2','LEU3','LEU4','LEU9','LYS1','LYS2','LYS4','LYS5','LYS9','LYS12','LYS20','LYS21','MET3','MET6','MET10','MET14','MET16','MET17','TRP2','TRP3','TRP4','TRP5','ILV1','ILV2','ILV3','ILV5','ILV6','ARO1','ARO2','ARO3','ARO4','ARO7','HOM2','HOM3','HOM6'],
    '脂质代谢': ['FAS1','FAS2','ACC1','OLE1','FAA1','FAA2','FAA3','FAA4','FAT1','POX1','FOX2','POT1','ERG1','ERG2','ERG3','ERG4','ERG5','ERG6','ERG7','ERG8','ERG9','ERG10','ERG11','ERG12','ERG13','ERG20','ERG24','ERG25','ERG26','ERG27'],
    '细胞周期': ['CDC2','CDC3','CDC4','CDC5','CDC6','CDC7','CDC8','CDC9','CDC10','CDC11','CDC12','CDC13','CDC14','CDC15','CDC16','CDC19','CDC20','CDC21','CDC23','CDC24','CDC25','CDC26','CDC27','CDC28','CDC31','CDC33','CDC34','CDC35','CDC36','CDC37','CDC39','CDC42','CDC43','CDC45','CDC46','CDC47','CDC48','CDC53','CDC55','CDC60'],
    '细胞壁/膜转运': ['PMA1','PMA2','VMA1','VMA2','VMA3','VMA4','VMA5','VMA6','VMA7','VMA8','VMA10','VMA13','VMA16','VPH1','VPH2','CHS1','CHS2','CHS3','CHS5','GSC2','FKS1','FKS2','GAS1','SED1','CWP1','CWP2','TIR1','TIR2','PIR1','PIR3','CCW12','CCW14'],
    '氧化还原/ROS': ['SOD1','SOD2','CTT1','CTA1','TSA1','TSA2','TRX1','TRX2','TRX3','TRR1','TRR2','GRX1','GRX2','GRX3','GRX4','GPX1','GPX2','PRX1','AHP1','HYR1','GLR1','GSH1','GSH2','ZWF1','SOL3','SOL4','GND1','GND2','TKL1','TKL2','TAL1','RPE1','RKI1'],
    '蛋白酶体/泛素': ['PRE1','PRE2','PRE3','PRE4','PRE5','PRE6','PRE7','PRE8','PRE9','PRE10','PUP1','PUP2','PUP3','SCL1','RPN1','RPN2','RPN3','RPN5','RPN6','RPN8','RPN9','RPN10','RPN11','RPN12','RPT1','RPT2','RPT3','RPT4','RPT5','RPT6','UBI4','UBC1','UBC4','UBC5','UBC6','UBC8','CDC34','RAD6'],
    '氨基酸转运': ['GAP1','AGP1','AGP2','AGP3','GNP1','BAP2','BAP3','TAT1','TAT2','DIP5','CAN1','ALP1','GAP1','HIP1','LYP1','PUT4','MUP1','MUP3','BAP2','BAP3','TAT1','TAT2','VAP1','YCT1','AGC1','GGC1','ODC1','ODC2','CIT2','CRC1','DIC1','ORT1','PET9','SFC1'],
    '转录调控': ['GCN4','GCN5','ADA2','ADA3','SPT3','SPT7','SPT8','SPT20','TAF1','TAF2','TAF3','TAF5','TAF6','TAF7','TAF8','TAF9','TAF10','TAF11','TAF12','TAF13','TAF14','SIN3','RPD3','SAP30','UME1','UME6','PHO23','SDS3','DEP1','HDA1','HDA2','HDA3','HOS1','HOS2','HOS3','SIR2'],
}

# ---------- Load model ----------
_s29 = importlib.util.spec_from_file_location('m29', r'D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\04_model_v29.py')
_m29 = importlib.util.module_from_spec(_s29); _s29.loader.exec_module(_m29)

m29 = _m29.VCellModel(feats, P=P).to(DEV)
m29.load_state_dict(torch.load(f'{DATA}/model_v29_best.pt', map_location=DEV, weights_only=True))
m29.set_strain_avg(); m29.eval()

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
    x={'bio':[torch.from_numpy(feats['strain_id'][idx]).to(DEV),torch.from_numpy(feats['chem_id'][idx]).to(DEV),
              torch.from_numpy(feats['chem_hash'][idx]).to(DEV),torch.from_numpy(feats['medium_onehot'][idx]).to(DEV),
              torch.from_numpy(feats['temp_norm'][idx]).to(DEV),torch.from_numpy(feats['time_feat'][idx]).to(DEV),
              torch.from_numpy(feats['sm_id'][idx]).to(DEV),torch.from_numpy(feats['ct_id'][idx]).to(DEV)],
       'ctx':[torch.from_numpy(feats['src_id'][idx]).to(DEV),torch.from_numpy(feats['ins_id'][idx]).to(DEV),
              torch.from_numpy(feats['plt_id'][idx]).to(DEV)],
       'seen':[torch.from_numpy(feats['chem_seen'][idx]).to(DEV),torch.from_numpy(feats['strain_seen'][idx]).to(DEV)],
       'ctx_prior':torch.from_numpy(ctx_all[idx]).to(DEV)}
    with torch.no_grad():
        return m29(x).cpu().numpy()

# ---------- 分析 ----------
print("=" * 70)
print("通路富集分析：预测 Δ 在已知功能模块中的差异显著性")
print("=" * 70)

for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
    idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    if len(idx) == 0: continue
    
    pred = predict(idx)
    yc = np.array([mc_mean(s) for s in idx])
    dp = pred - yc
    abs_dp = np.nanmean(np.abs(dp), axis=0)
    
    print(f"\n--- {scene} ({len(idx)} 样本) ---")
    
    results = []
    for pw_name, genes in PATHWAYS.items():
        indices = [gene2i[g] for g in genes if g in gene2i]
        if len(indices) < 3: continue
        
        in_path = abs_dp[indices]
        out_path = abs_dp[[i for i in range(P) if i not in indices]]
        in_path = in_path[np.isfinite(in_path)]
        out_path = out_path[np.isfinite(out_path)]
        if len(in_path) < 3: continue
        
        stat, pval = mannwhitneyu(in_path, out_path, alternative='greater')
        mean_in = np.mean(in_path)
        mean_out = np.mean(out_path)
        fold = mean_in / max(mean_out, 1e-8)
        results.append((pw_name, len(genes), len(indices), mean_in, fold, pval))
    
    results.sort(key=lambda x: x[5])
    
    print(f"  {'通路':<18} {'基因数':>5} {'匹配数':>5} {'|Δ|均值':>8} {'富集倍数':>7} {'P-value':>8}")
    for pw, n_g, n_m, mean_d, fold, pval in results:
        sig = "***" if pval<0.001 else "**" if pval<0.01 else "*" if pval<0.05 else ""
        if sig:
            print(f"  {pw:<18} {n_g:>5} {n_m:>5} {mean_d:>8.3f} {fold:>7.2f} {pval:>8.2e} {sig}")

# ---------- 高效应蛋白 ----------
print()
print("=" * 70)
print("全局 |Δ| Top 20 蛋白")
print("=" * 70)

# 用所有val场景的平均|Δ|
all_dps = []
for scene in ['val_chem_only', 'val_strain_only', 'val_both', 'val_time']:
    idx = np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    if len(idx)==0: continue
    pred = predict(idx)
    yc = np.array([mc_mean(s) for s in idx])
    all_dps.append(pred - yc)
all_dp = np.nanmean(np.concatenate(all_dps, axis=0), axis=0)
abs_all = np.abs(all_dp)
top20 = np.argsort(abs_all)[-20:][::-1]

print(f"  {'排名':<4} {'蛋白':<10} {'|Δ|':>7}  功能分类")
for rank, pi in enumerate(top20, 1):
    g = prot_names[pi]
    funcs = []
    for pw, genes in PATHWAYS.items():
        if g in genes:
            funcs.append(pw)
    print(f"  {rank:<4} {g:<10} {abs_all[pi]:>7.3f}  {'; '.join(funcs) if funcs else '未知'}")

print("\nDONE")
