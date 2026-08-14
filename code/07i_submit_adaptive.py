# -*- coding: utf-8 -*-
"""场景自适应提交：每个场景用各自最强的模型"""
import numpy as np, pandas as pd, pickle, torch, importlib.util, hashlib

DATA = 'data'; BASE = '../input'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float32)
mask = np.load(f'{DATA}/mask.npy').astype(np.float32)
feats = pickle.load(open(f'{DATA}/feats.pkl','rb'))
P = y_log2.shape[1]
ctx_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)
morgan_all = feats['chem_morgan']
prot_names = [l.strip() for l in open(f'{DATA}/prot_names.txt')]

_s21 = importlib.util.spec_from_file_location('m21', '04_model_v21.py')
_m21 = importlib.util.module_from_spec(_s21); _s21.loader.exec_module(_m21)
_s37 = importlib.util.spec_from_file_location('m37', '04_model_v37.py')
_m37 = importlib.util.module_from_spec(_s37); _s37.loader.exec_module(_m37)

def load(p, cls, sa=False):
    m = cls(feats, P=P); m.load_state_dict(torch.load(f'{DATA}/{p}', map_location=DEV, weights_only=True))
    if sa: m.set_strain_avg()
    return m.to(DEV).eval()

m21a = load('model_v21.pt', _m21.VCellModel)
m21b = load('model_v21_s43.pt', _m21.VCellModel)
m21c = load('model_v21_s44.pt', _m21.VCellModel)
m37 = load('model_v37_best.pt', _m37.VCellModel, sa=True)

def pred_with(idx, models):
    x = {'bio':[torch.from_numpy(feats['strain_id'][idx]),torch.from_numpy(feats['chem_id'][idx]),
                torch.from_numpy(feats['chem_hash'][idx]),torch.from_numpy(feats['medium_onehot'][idx]),
                torch.from_numpy(feats['temp_norm'][idx]),torch.from_numpy(feats['time_feat'][idx]),
                torch.from_numpy(feats['sm_id'][idx]),torch.from_numpy(feats['ct_id'][idx])],
         'ctx':[torch.from_numpy(feats['src_id'][idx]),torch.from_numpy(feats['ins_id'][idx]),torch.from_numpy(feats['plt_id'][idx])],
         'seen':[torch.from_numpy(feats['chem_seen'][idx]),torch.from_numpy(feats['strain_seen'][idx])],
         'ctx_prior':torch.from_numpy(ctx_all[idx]),'chem_morgan':torch.from_numpy(morgan_all[idx])}
    preds=[]
    with torch.no_grad():
        for m in models:
            xg={k:(v.to(DEV) if k in ('ctx_prior','chem_morgan') else [t.to(DEV) for t in v]) for k,v in x.items()}
            preds.append(m(xg).cpu().numpy())
    return np.mean(preds,axis=0)

ENS21 = [m21a, m21b, m21c]
# 场景 -> 模型选择（val_time 用 v3.7，因为 0.829 > v2.1 的 0.826）
SCENE_MODEL = {
    'val_chem_only': ENS21, 'val_strain_only': [m37], 'val_both': [m37], 'val_time': [m37],
    'test_chem_only': ENS21, 'test_strain_only': [m37], 'test_both': [m37], 'test_time': [m37],
}

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

print("=== val 场景自适应验证 ===")
print(f"{'场景':<18}{'自适应R2':>9}{'v3.7单':>9}{'v2.1集成':>9}")
print('-'*50)
for scene in ['val_chem_only','val_strain_only','val_both','val_time']:
    idx=np.where(meta['split_final'].eq(scene).values & meta['role'].eq('treatment').values)[0]
    yt=y_log2[idx]; m=mask[idx].astype(bool)
    p_adapt = pred_with(idx, SCENE_MODEL[scene])
    p37 = pred_with(idx, [m37])
    p21 = pred_with(idx, ENS21)
    r_adapt = prot_r2(np.where(m,p_adapt,0),np.where(m,yt,0),m)
    r37 = prot_r2(np.where(m,p37,0),np.where(m,yt,0),m)
    r21 = prot_r2(np.where(m,p21,0),np.where(m,yt,0),m)
    print(f"{scene:<18}{r_adapt:>9.3f}{r37:>9.3f}{r21:>9.3f}")

# test 提交
tmeta = pd.read_csv(f'{BASE}/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')
strains_tr = sorted(meta['Strains'].unique()); chems_tr = sorted(meta.loc[meta['role'].eq('treatment'), 'perturbation_no_concentration'].unique())
strain2id = {s:i for i,s in enumerate(strains_tr)}; chem2id = {c:i for i,c in enumerate(chems_tr)}
sm_cats = sorted((meta['Strains'].astype(str)+'|'+meta['Medium'].astype(str)).unique())
ct_cats = sorted((meta['perturbation_no_concentration'].astype(str)+'|'+meta['Temperature'].astype(str)).unique())
sm2id={k:i for i,k in enumerate(sm_cats)}; ct2id={k:i for i,k in enumerate(ct_cats)}
src_cats=sorted(meta['data_source'].unique()); ins_cats=sorted(meta['instrument'].unique()); plt_cats=sorted(meta['Yeast_cell_plate'].unique())
src2id={k:i for i,k in enumerate(src_cats)}; ins2id={k:i for i,k in enumerate(ins_cats)}; plt2id={k:i for i,k in enumerate(plt_cats)}
train_strains=set(meta.loc[meta['split_final'].eq('train'),'Strains'])
train_chems=set(meta.loc[meta['split_final'].eq('train') & meta['role'].eq('treatment'),'perturbation_no_concentration'])
def hash_vec(name,dim=32):
    h=hashlib.sha256(str(name).encode()).hexdigest()
    return np.array([int(h[i*2:i*2+2],16)/255.0 for i in range(dim)])
tstrain=np.array([strain2id.get(s,-1) for s in tmeta['Strains']],dtype=np.int64)
tchem=np.array([chem2id.get(c,-1) for c in tmeta['perturbation_no_concentration']],dtype=np.int64)
tmed=np.array([[1.0 if m==mm else 0.0 for mm in sorted(meta['Medium'].unique())] for m in tmeta['Medium']],dtype=np.float32)
ttemp=((tmeta['Temperature'].astype(float)-30.0)/7.0).values.astype(np.float32)
tt=tmeta['pert_time'].astype(float).values; tt_log=np.log2(tt/15.0)/np.log2(240.0/15.0)
ttfeat=np.stack([tt_log,np.sin(2*np.pi*tt_log),np.cos(2*np.pi*tt_log)],axis=1).astype(np.float32)
tsm=np.array([sm2id.get(f"{s}|{m}",-1) for s,m in zip(tmeta['Strains'],tmeta['Medium'])],dtype=np.int64)
tct=np.array([ct2id.get(f"{c}|{t_}",-1) for c,t_ in zip(tmeta['perturbation_no_concentration'],tmeta['Temperature'])],dtype=np.int64)
tsrc=np.array([src2id.get(s,-1) for s in tmeta['data_source']],dtype=np.int64)
tins=np.array([ins2id.get(s,-1) for s in tmeta['instrument']],dtype=np.int64)
tplt=np.array([plt2id.get(s,-1) for s in tmeta['Yeast_cell_plate']],dtype=np.int64)
tchash=np.array([hash_vec(c) for c in tmeta['perturbation_no_concentration']],dtype=np.float32)
tcseen=np.array([1.0 if c in train_chems else 0.0 for c in tmeta['perturbation_no_concentration']],dtype=np.float32)
tsseen=np.array([1.0 if s in train_strains else 0.0 for s in tmeta['Strains']],dtype=np.float32)
# test morgan
pert2morgan={}
for i,p in enumerate(meta['perturbation_no_concentration'].values):
    pert2morgan.setdefault(p, morgan_all[i])
tmorgan=np.array([pert2morgan.get(c, np.zeros(64,dtype=np.float32)) for c in tmeta['perturbation_no_concentration']],dtype=np.float32)
# test ctx_prior
gmean=feats['gmean']
ctx_key_tr=(meta['Strains'].astype(str)+'|'+meta['Medium'].astype(str)+'|'+meta['Temperature'].astype(str)+'|'+meta['pert_time'].astype(str)).values
train_mask_arr=meta['split_final'].eq('train').values
t_ctx_prior=np.tile(gmean,(len(tmeta),1)).astype(np.float32)
ctx_grp={}
for i in np.where(train_mask_arr)[0]:
    ctx_grp.setdefault(ctx_key_tr[i],[]).append(tr_y_nan[i])
for k,vals in ctx_grp.items(): ctx_grp[k]=np.nanmean(vals,axis=0)
t_ctx=(tmeta['Strains'].astype(str)+'|'+tmeta['Medium'].astype(str)+'|'+tmeta['Temperature'].astype(str)+'|'+tmeta['pert_time'].astype(str)).values
for i,k in enumerate(t_ctx):
    if k in ctx_grp: t_ctx_prior[i]=ctx_grp[k]
t_ctx_prior=np.nan_to_num(t_ctx_prior,nan=0.0)

def tpred(models):
    x={'bio':[torch.from_numpy(tstrain),torch.from_numpy(tchem),torch.from_numpy(tchash),
              torch.from_numpy(tmed),torch.from_numpy(ttemp),torch.from_numpy(ttfeat),
              torch.from_numpy(tsm),torch.from_numpy(tct)],
       'ctx':[torch.from_numpy(tsrc),torch.from_numpy(tins),torch.from_numpy(tplt)],
       'seen':[torch.from_numpy(tcseen),torch.from_numpy(tsseen)],
       'ctx_prior':torch.from_numpy(t_ctx_prior),'chem_morgan':torch.from_numpy(tmorgan)}
    preds=[]
    with torch.no_grad():
        for m in models:
            xg={k:(v.to(DEV) if k in ('ctx_prior','chem_morgan') else [t.to(DEV) for t in v]) for k,v in x.items()}
            preds.append(m(xg).cpu().numpy())
    return np.mean(preds,axis=0)

# 分场景预测
pred = np.zeros((len(tmeta), P), dtype=np.float32)
for scene, models in [('test_chem_only',ENS21),('test_strain_only',[m37]),('test_both',[m37]),('test_time',ENS21)]:
    idx = np.where(tmeta['split_final'].eq(scene).values)[0]
    if len(idx)==0: continue
    # 用全量特征预测，再取对应行的结果（简化：直接全量预测各模型，再按行取）
    pass

# 更简单：全量预测两个方案，按场景拼
p21_full = tpred(ENS21)
p37_full = tpred([m37])
for scene in ['test_chem_only','test_strain_only','test_both','test_time']:
    idx = np.where(tmeta['split_final'].eq(scene).values)[0]
    if scene == 'test_chem_only':
        pred[idx] = p21_full[idx]
    else:
        pred[idx] = p37_full[idx]

sub = pd.DataFrame(pred, index=tmeta.index, columns=prot_names)
sub.index.name='sample_ID'
sub.to_csv(f'{DATA}/prediction_adaptive.csv')
assert len(sub)==4454 and sub.shape[1]==4422
assert not sub.isna().any().any() and np.isfinite(sub.values).all()
print("\n[提交] prediction_adaptive.csv 生成并校验通过")
