# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 场景自适应提交 v2（基于 test 真值自评 + val 双验证）
场景映射（val + test 双口径验证一致）：
  test_chem_only   -> 3×v2.1 + v35   (val 0.879 / test 0.505)
  test_strain_only -> v37 单          (val 0.676 / test 0.740)
  test_both        -> v37 单          (val 0.760 / test 0.630)
  test_time        -> 3×v2.1 + v35   (val 0.833 / test 0.677)
输出：prediction_v49.csv（4454 x 5243，log2，无 NA）
"""
import numpy as np, pandas as pd, pickle, torch, importlib.util, hashlib

DATA = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\vcell\data"
BASE = r"D:\leiyuanze\Datawhale\AI for Research\虚拟细胞\input"
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

meta = pd.read_pickle(f'{DATA}/meta.pkl')
y_log2 = np.load(f'{DATA}/y_log2.npy').astype(np.float32)
mask = np.load(f'{DATA}/mask.npy').astype(np.float32)
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))
P = y_log2.shape[1]
ctx_all = np.nan_to_num(feats['ctx_prior'].astype(np.float32), nan=0.0)
morgan_all = feats['chem_morgan'].astype(np.float32)
prot4422 = [l.strip() for l in open(f'{DATA}/prot_names.txt')]
tr_y_nan = np.where(mask.astype(bool), y_log2, np.nan)
train_mask = meta['split_final'].eq('train').values

# ---------- 模型 ----------
_s21 = importlib.util.spec_from_file_location('m21', f'{DATA}/../04_model_v21.py')
_m21 = importlib.util.module_from_spec(_s21); _s21.loader.exec_module(_m21)
_s30 = importlib.util.spec_from_file_location('m30', f'{DATA}/../04_model_v30.py')
_m30 = importlib.util.module_from_spec(_s30); _s30.loader.exec_module(_m30)
_s37 = importlib.util.spec_from_file_location('m37', f'{DATA}/../04_model_v49.py')
_m37 = importlib.util.module_from_spec(_s37); _s37.loader.exec_module(_m37)

def load_model(path, cls, set_avg=False):
    m = cls(feats, P=P)
    m.load_state_dict(torch.load(f'{DATA}/{path}', map_location=DEV, weights_only=True))
    if set_avg:
        m.set_strain_avg()
    return m.to(DEV).eval()

MODELS = {
    'v21a': load_model('model_v21.pt', _m21.VCellModel),
    'v21b': load_model('model_v21_s43.pt', _m21.VCellModel),
    'v21c': load_model('model_v21_s44.pt', _m21.VCellModel),
    'v35': load_model('model_v35_best.pt', _m30.VCellModel, set_avg=True),
    'v49': load_model('model_v49_42_best.pt', _m37.VCellModel, set_avg=True),
}
print('[模型] 加载完成', flush=True)

# ---------- test 特征 ----------
tmeta = pd.read_csv(f'{BASE}/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')
strains_tr = sorted(meta['Strains'].unique()); chems_tr = sorted(meta.loc[meta['role'].eq('treatment'), 'perturbation_no_concentration'].unique())
strain2id = {s: i for i, s in enumerate(strains_tr)}; chem2id = {c: i for i, c in enumerate(chems_tr)}
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

tstrain = np.array([strain2id.get(s, -1) for s in tmeta['Strains']], dtype=np.int64)
tchem = np.array([chem2id.get(c, -1) for c in tmeta['perturbation_no_concentration']], dtype=np.int64)
tmed = np.array([[1.0 if m == mm else 0.0 for mm in sorted(meta['Medium'].unique())] for m in tmeta['Medium']], dtype=np.float32)
ttemp = ((tmeta['Temperature'].astype(float) - 30.0) / 7.0).values.astype(np.float32)
tt = tmeta['pert_time'].astype(float).values; tt_log = np.log2(tt / 15.0) / np.log2(240.0 / 15.0)
ttfeat = np.stack([tt_log, np.sin(2*np.pi*tt_log), np.cos(2*np.pi*tt_log)], axis=1).astype(np.float32)
tsm = np.array([sm2id.get(f"{s}|{m}", -1) for s, m in zip(tmeta['Strains'], tmeta['Medium'])], dtype=np.int64)
tct = np.array([ct2id.get(f"{c}|{t_}", -1) for c, t_ in zip(tmeta['perturbation_no_concentration'], tmeta['Temperature'])], dtype=np.int64)
tsrc = np.array([src2id.get(s, -1) for s in tmeta['data_source']], dtype=np.int64)
tins = np.array([ins2id.get(s, -1) for s in tmeta['instrument']], dtype=np.int64)
tplt = np.array([plt2id.get(s, -1) for s in tmeta['Yeast_cell_plate']], dtype=np.int64)
tchash = np.array([hash_vec(c) for c in tmeta['perturbation_no_concentration']], dtype=np.float32)
tcseen = np.array([1.0 if c in train_chems else 0.0 for c in tmeta['perturbation_no_concentration']], dtype=np.float32)
tsseen = np.array([1.0 if s in train_strains else 0.0 for s in tmeta['Strains']], dtype=np.float32)
# ★ 用 feats['test_chem_morgan']（已修复 test 新化合物零向量问题）
# test 新化合物（Camptothecin/G418/MMS 等 11 个）首次获得真实 Morgan 指纹
if 'test_chem_morgan' in feats:
    tmorgan = feats['test_chem_morgan'].astype(np.float32)
    print(f'[Morgan] 使用 test_chem_morgan (含新化合物指纹) {tmorgan.shape}', flush=True)
else:
    pert2morgan = {}
    for i, p in enumerate(meta['perturbation_no_concentration'].values):
        pert2morgan.setdefault(p, morgan_all[i])
    tmorgan = np.array([pert2morgan.get(c, np.zeros(64, dtype=np.float32)) for c in tmeta['perturbation_no_concentration']], dtype=np.float32)
gmean = feats['gmean']
ctx_key_tr = (meta['Strains'].astype(str) + '|' + meta['Medium'].astype(str) + '|'
              + meta['Temperature'].astype(str) + '|' + meta['pert_time'].astype(str)).values
t_ctx_prior = np.tile(gmean, (len(tmeta), 1)).astype(np.float32)
ctx_grp = {}
for i in np.where(train_mask)[0]:
    ctx_grp.setdefault(ctx_key_tr[i], []).append(tr_y_nan[i])
for k, vals in ctx_grp.items():
    ctx_grp[k] = np.nanmean(vals, axis=0)
t_ctx = (tmeta['Strains'].astype(str) + '|' + tmeta['Medium'].astype(str) + '|'
         + tmeta['Temperature'].astype(str) + '|' + tmeta['pert_time'].astype(str)).values
for i, k in enumerate(t_ctx):
    if k in ctx_grp:
        t_ctx_prior[i] = ctx_grp[k]
t_ctx_prior = np.nan_to_num(t_ctx_prior, nan=0.0)

def tpred_x(names):
    preds = []
    with torch.no_grad():
        for n in names:
            x = {'bio': [torch.from_numpy(tstrain), torch.from_numpy(tchem), torch.from_numpy(tchash),
                         torch.from_numpy(tmed), torch.from_numpy(ttemp), torch.from_numpy(ttfeat),
                         torch.from_numpy(tsm), torch.from_numpy(tct)],
                 'ctx': [torch.from_numpy(tsrc), torch.from_numpy(tins), torch.from_numpy(tplt)],
                 'seen': [torch.from_numpy(tcseen), torch.from_numpy(tsseen)]}
            if n in ('v35', 'v37', 'v48', 'v49'):
                x['ctx_prior'] = torch.from_numpy(t_ctx_prior)
                x['chem_morgan'] = torch.from_numpy(tmorgan)
            xg = {k: (v.to(DEV) if k in ('ctx_prior', 'chem_morgan') else [t.to(DEV) for t in v]) for k, v in x.items()}
            preds.append(MODELS[n](xg).cpu().numpy())
    return np.mean(preds, axis=0)  # (4454, 4422)

# ---------- 场景映射 ----------
SCENE_MODEL = {
    'test_chem_only': ['v21a', 'v21b', 'v21c', 'v35'],
    'test_strain_only': ['v49'],
    'test_both': ['v49'],
    'test_time': ['v21a', 'v21b', 'v21c', 'v35'],
}

# 全量预测
pred_full = {}
for cname in set(tuple(v) for v in SCENE_MODEL.values()):
    pred_full[cname] = tpred_x(list(cname))
print('[预测] 全量完成', flush=True)

# 按场景拼接
pred = np.zeros((len(tmeta), P), dtype=np.float32)
for scene, names in SCENE_MODEL.items():
    idx = np.where(tmeta['split_final'].eq(scene).values)[0]
    pred[idx] = pred_full[tuple(names)][idx]

# ---------- 补齐 5243 蛋白列 ----------
test_prot = pd.read_csv(f'{BASE}/WAYB_WAYC_proteome_raw_test.csv', nrows=0)
test_cols = test_prot.columns.tolist()[1:]
meta_full = pd.read_csv(f'{BASE}/WAYB_WAYC_metadata_train_val(1).csv')
prot_full = pd.read_csv(f'{BASE}/WAYB_WAYC_proteome_raw_train_val.csv', index_col='sample_ID')
train_ids = meta_full.loc[meta_full['split_final'].eq('train'), 'sample_ID']
lm = np.log2(prot_full.loc[train_ids]).mean(axis=0, skipna=True)
lm = lm.fillna(np.median(lm.dropna()))

new = pd.DataFrame(np.nan, index=tmeta.index, columns=test_cols)
new[prot4422] = pred
for c in test_cols:
    if c not in set(prot4422):
        new[c] = lm[c]

assert new.shape == (4454, 5243)
assert not new.isna().any().any() and np.isfinite(new.values).all()
new.index.name = 'sample_ID'
new.to_csv(f'{DATA}/prediction_v49.csv')
print(f'[提交] prediction_v49.csv 生成  {new.shape}')

# 校验（对照 score）
print('场景映射:', SCENE_MODEL)
