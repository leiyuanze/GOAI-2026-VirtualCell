# -*- coding: utf-8 -*-
"""
给 test 新化合物补 SMILES + Morgan 指纹（重大空白修复）
test_chem_only / test_both 的新化合物从未有结构指纹（feats 里全零向量），
导致 C 分支对这些化合物只能输出零。本脚本：
1. 列出所有 train_val 未出现但 test 出现的化合物
2. 从 PubChem 下载 SMILES（名称映射，复用原脚本的映射表）
3. 用 RDKit 生成 Morgan 指纹（2048 bit -> PCA 到 64 维，与现有 pipeline 一致）
4. 更新 feats.pkl：新增 test_chem_morgan（test 行索引的指纹）
"""
import json, pickle, time, urllib.request, urllib.parse
import numpy as np
import pandas as pd

DATA = 'data'
meta = pd.read_pickle(f'{DATA}/meta.pkl')
tmeta = pd.read_csv('../input/WAYB_WAYC_metadata_test(1).csv').set_index('sample_ID')
feats = pickle.load(open(f'{DATA}/feats.pkl', 'rb'))

# 1. 找出 test 出现但 train_val 没有的化合物
train_chems = set(meta['perturbation_no_concentration'].unique())
test_chems = sorted(tmeta['perturbation_no_concentration'].unique())
new_chems = [c for c in test_chems if c not in train_chems]
print(f'test 出现但 train_val 没有的化合物: {len(new_chems)}')
for c in new_chems:
    print(f'  {c}')

# 2. PubChem 查询（复用原脚本映射表）
NAME_MAP = {
    'CHX': 'Cycloheximide', 'FCCP': 'Carbonyl cyanide 4-(trifluoromethoxy)phenylhydrazone',
    'EDTA': 'Edetic acid', 'SDS': 'Sodium dodecyl sulfate', 'NaCl': 'Sodium chloride',
    '(1R, 2S, 5R) - (-) - Menthol': 'Menthol', '1-10 Phenanthroline monohydrate': '1,10-Phenanthroline',
    'LY 294002 hydrochloride': 'LY294002', 'Amiodarone hydrochloride': 'Amiodarone',
    'Clomiphene citrate': 'Clomiphene', 'Desipramine hydrochloride': 'Desipramine',
    'Dyclonine hydrochloride': 'Dyclonine', 'Harmine hydrochloride': 'Harmine',
    'Nystatin dihydrate': 'Nystatin', 'Pentamidine isethionate': 'Pentamidine',
    'Raloxifene hydrochloride': 'Raloxifene', 'Trifluoperazine dihydrochloride': 'Trifluoperazine',
    'Rapamycin': 'Sirolimus',
    # 新化合物映射
    '(S)-(+)-Camptothecin': 'Camptothecin', 'Abietic acid': 'Abietic acid',
    'Doxycycline hyclate': 'Doxycycline', 'Fluconazole': 'Fluconazole',
    'G418': 'Geneticin', 'H2O2': 'Hydrogen peroxide', 'Hygromycin B': 'Hygromycin B',
    'MMS': 'Methyl methanesulfonate', 'Neomycin B': 'Neomycin',
    'Plumbagin': 'Plumbagin', 'Tamoxifen': 'Tamoxifen',
}

def pubchem_smiles(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(name)}/property/IsomericSMILES/JSON"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    r = urllib.request.urlopen(req, timeout=30)
    data = json.loads(r.read().decode('utf-8'))
    props = data['PropertyTable']['Properties'][0]
    for k in ('SMILES', 'IsomericSMILES', 'CanonicalSMILES', 'ConnectivitySMILES'):
        if k in props and props[k]:
            return props[k], props.get('CID')
    return None, props.get('CID')

result = {}
failed = []
for c in new_chems:
    qname = NAME_MAP.get(c, c)
    done = False
    for name in [qname, c]:
        try:
            smi, cid = pubchem_smiles(name)
            result[c] = {'query_name': name, 'cid': cid, 'smiles': smi}
            print(f"[OK] {c} -> {smi[:60] if smi else 'NO_SMILES'}")
            done = True
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            failed.append(c)
            print(f"[HTTP {e.code}] {c} ({name})")
            done = True
            break
        except Exception as e:
            if name == c:
                failed.append(c)
                print(f"[FAIL] {c} ({qname}): {type(e).__name__}")
                done = True
    if not done:
        failed.append(c)
        print(f"[FAIL] {c}: 所有名称查不到")
    time.sleep(0.3)

print(f"\n成功 {len(result)}/{len(new_chems)}，失败 {len(failed)}")
if failed:
    print('失败:', failed)

# 合并到 chem_smiles.json
smiles_path = f'{DATA}/chem_smiles.json'
try:
    with open(smiles_path, encoding='utf-8') as f:
        old = json.load(f)
except Exception:
    old = {}
old.update(result)
with open(smiles_path, 'w', encoding='utf-8') as f:
    json.dump(old, f, indent=2, ensure_ascii=False)
print(f'已更新 {smiles_path}，共 {len(old)} 化合物')
