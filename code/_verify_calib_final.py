# -*- coding: utf-8 -*-
"""验证后处理校准：pred_calib = pred - control_mean + global_mean 对四场景的影响"""
import numpy as np, pandas as pd, pickle, torch, importlib.util

DATA = 'data'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float32)
mask = np.load(f'{DATA}/mask.npy').astype(np.float32)
feats = pickle.load(open(f'{DATA}/feats.pkl','rb'))
P = y_log2.shape[1]
ctx_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)
morgan_all = feats['chem_morgan']
gmean = feats['gmean']

_s21 = importlib.util.spec_from_file_location('m21', '04_model_v21.py')
_m21 = importlib.util.module_from_spec(_s21); _s21.loader.exec_module(_m21)
_s30 = importlib.util.spec_from_file_location('m30', '04_model_v30.py')
_m30 = importlib.util.module_from_spec(_s30); _s30.loader.exec_module(_m30)

def load(p, cls, sa=False):
    m = cls(feats, P=P); m.load_state_dict(torch.load(f'{DATA}/{p}', map_location=DEV, weights_only=True))
    if sa: m.set_strain_avg()
    return m.to(DEV).eval()

models = [(load('model_v21.pt',_m21.VCellModel),'v21'),(load('model_v21_s43.pt',_m21.VCellModel),'v21'),
          (load('model_v21_s44.pt',_m21.VCellModel),'v21'),(load('model_v35_best.pt',_m30.VCellModel,sa=True),'v30')]

def predict(idx):
    x = {'bio':[torch.from_numpy(feats['strain_id'][idx]),torch.from_numpy(feats['chem_id'][idx]),
                torch.from_numpy(feats['chem_hash'][idx]),torch.from_numpy(feats['medium_onehot'][idx]),
                torch.from_numpy(feats['temp_norm'][idx]),torch.from_numpy(feats['time_feat'][idx]),
                torch.from_numpy(feats['sm_id'][idx]),torch.from_numpy(feats['ct_id'][idx])],
         'ctx':[torch.from_numpy(feats['src_id'][idx]),torch.from_numpy(feats['ins_id'][idx]),torch.from_numpy(feats['plt_id'][idx])],
         'seen':[torch.from_numpy(feats['chem_seen'][idx]),torch.from_numpy(feats['strain_seen'][idx])],
         'ctx_prior':torch.from_numpy(ctx_all[idx]),'chem_morgan':torch.from_numpy(morgan_all[idx])}
    preds=[]
    with torch.no_grad():
        for m,tag in models:
            xg={k:(v.to(DEV) if k in ('ctx_prior','chem_morgan') else [t.to(DEV) for t in v]) for k,v in x.items()}
            preds.append(m(xg).cpu().numpy())
    return np.mean(preds,axis=0)

# matched control
ctrl_idx = np.where(meta['role'].eq('control').values)[0]
ctrl_key = (meta.iloc[ctrl_idx]['data_source'].astype(str)+'|'+meta.iloc[ctrl_idx]['instrument'].astype(str)+'|'
            +meta.iloc[ctrl_idx]['Yeast_cell_plate'].astype(str)+'|'+meta.iloc[ctrl_idx]['Strains'].astype(str)+'|'
            +meta.iloc[ctrl_idx]['Medium'].astype(str)+'|'+meta.iloc[ctrl_idx]['Temperature'].astype(str)+'|'
            +meta.iloc[ctrl_idx]['pert_time'].astype(str)).values
ctrl_lookup={}
for k,pos in zip(ctrl_key,ctrl_idx): ctrl_lookup.setdefault(k,[]).append(pos)
tr_y_nan = np.where(mask.astype(bool), y_log2, np.nan)
def mc(sid):
    r=meta.iloc[sid]
    k=(str(r['data_source'])+'|'+str(r['instrument'])+'|'+str(r['Yeast_cell_plate'])+'|'
       +str(r['Strains'])+'|'+str(r['Medium'])+'|'+str(r['Temperature'])+'|'+str(r['pert_time']))
    if k not in ctrl_lookup: return None
    rows=ctrl_lookup[k]; cvals=tr_y_nan[rows]; cm=mask[rows]>0
    with np.errstate(invalid='ignore'):
        return np.where(cm.sum(0)>0, np.nansum(np.where(cm,cvals,np.nan),0)/cm.sum(0), np.nan)

def prot_r2(yp,yt,m):
    cnt=m.sum(0); n=np.maximum(cnt.astype(float),1)
    ytc=np.where(m,yt,0.0); ypc=np.where(m,yp,0.0)
    mt=ytc.sum(0)/n
    ss_tot=(((ytc-mt)**2)*m).sum(0); ss_res=(((ytc-ypc)**2)*m).sum(0)
    return float(np.median(1-ss_res/np.maximum(ss_tot,1e-12)))

print(f"{'场景':<18}{'蛋白R2原':>9}{'蛋白R2校准':>10}{'FC原':>8}{'FC校准':>8}")
print('-'*60)
for scene in ['val_chem_only','val_strain_only','val_both','val_time']:
    idx=np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    pred=predict(idx)
    yt=y_log2[idx]; m=mask[idx].astype(bool)
    yc=np.array([mc(s) for s in idx])
    # 校准：pred - control + global
    yc_fill = np.where(np.isfinite(yc), yc, gmean)
    pred_calib = pred - yc_fill + gmean
    r2_raw = prot_r2(np.where(m,pred,0), np.where(m,yt,0), m)
    r2_cal = prot_r2(np.where(m,pred_calib,0), np.where(m,yt,0), m)
    def fc(yp):
        ok=np.isfinite(yc)&m&np.isfinite(yp)
        if ok.sum()<10: return float('nan')
        dp=(yp-yc)[ok]; dt=(yt-yc)[ok]
        return float(np.corrcoef(dp,dt)[0,1])
    print(f"{scene:<18}{r2_raw:>9.3f}{r2_cal:>10.3f}{fc(pred):>8.3f}{fc(pred_calib):>8.3f}")
