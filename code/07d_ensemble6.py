# -*- coding: utf-8 -*-
"""07d 集成 3×v2.1 + v2.9 评估"""
import numpy as np, pandas as pd, pickle, torch, importlib.util

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float32)
mask = np.load(f'{DATA}/mask.npy').astype(np.float32)
feats = pickle.load(open(f'{DATA}/feats.pkl','rb'))
P = y_log2.shape[1]
ctx_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)

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
tr_y_nan=np.where(mask.astype(bool),y_log2,np.nan)

def mc_mean(sid):
    r=meta.iloc[sid]
    k=(str(r['data_source'])+'|'+str(r['instrument'])+'|'+str(r['Yeast_cell_plate'])+'|'
       +str(r['Strains'])+'|'+str(r['Medium'])+'|'+str(r['Temperature'])+'|'+str(r['pert_time']))
    if k not in ctrl_lookup: return None
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

print(f"新集成 3×v2.1+v2.9 | {'场景':<16}{'样本':>5}{'RMSE':>7}{'GlobalR2':>9}{'蛋白R2中位':>10}{'FC PCC':>8}")
print("-"*65)
for scene in ['val_chem_only','val_strain_only','val_both','val_time']:
    idx=np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    if len(idx)==0: continue
    pred=predict(idx)
    yt,m=y_log2[idx],mask[idx].astype(bool)
    valid=m & np.isfinite(pred)
    rmse=float(np.sqrt(((yt[m]-pred[m])**2).mean()))
    a,b=yt[valid],pred[valid]
    g2=1-((a-b)**2).sum()/max(((a-a.mean())**2).sum(),1e-12)
    cnt=valid.sum(0);keep=cnt>=3;n=np.maximum(cnt.astype(float),1)
    ytc=np.where(valid,yt,0.0);ypc=np.where(valid,pred,0.0)
    mt=ytc.sum(0)/n
    ss_tot=(((ytc-mt)**2)*valid).sum(0);ss_res=(((ytc-ypc)**2)*valid).sum(0)
    p2=float(np.median(1-ss_res/np.maximum(ss_tot,1e-12)))
    yc=np.array([mc_mean(s) for s in idx])
    fc_ok=np.isfinite(yc)&m&np.isfinite(pred)
    dp=(pred-yc)[fc_ok];dt=(yt-yc)[fc_ok]
    fc=float(np.corrcoef(dp,dt)[0,1]) if len(dp)>10 else float('nan')
    print(f"{scene:<20}{len(idx):>5}{rmse:>7.3f}{g2:>9.3f}{p2:>10.3f}{fc:>8.3f}")

print("\nDONE")
