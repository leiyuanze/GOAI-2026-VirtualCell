# -*- coding: utf-8 -*-
"""
从 PubChem 下载 43 种化合物的 SMILES（IsomericSMILES，含立体化学）
输出 chem_smiles.json
"""
import json
import time
import urllib.request
import urllib.parse

NAME_MAP = {
    'CHX': 'Cycloheximide',
    'FCCP': 'Carbonyl cyanide 4-(trifluoromethoxy)phenylhydrazone',
    'EDTA': 'Edetic acid',
    'SDS': 'Sodium dodecyl sulfate',
    'NaCl': 'Sodium chloride',
    '(1R, 2S, 5R) - (-) - Menthol': 'Menthol',
    '1-10 Phenanthroline monohydrate': '1,10-Phenanthroline',
    'LY 294002 hydrochloride': 'LY294002',
    'Amiodarone hydrochloride': 'Amiodarone',
    'Clomiphene citrate': 'Clomiphene',
    'Desipramine hydrochloride': 'Desipramine',
    'Dyclonine hydrochloride': 'Dyclonine',
    'Harmine hydrochloride': 'Harmine',
    'Nystatin dihydrate': 'Nystatin',
    'Pentamidine isethionate': 'Pentamidine',
    'Raloxifene hydrochloride': 'Raloxifene',
    'Trifluoperazine dihydrochloride': 'Trifluoperazine',
    'Rapamycin': 'Sirolimus',
}

COMPOUNDS = [
    '(1R, 2S, 5R) - (-) - Menthol', '1-10 Phenanthroline monohydrate', '4-Hydroxytamoxifen',
    'Amiodarone hydrochloride', 'Amphotericin B', 'Anisomycin', 'Artemisinin', 'Brefeldin A',
    'CHX', 'Cisplatin', 'Clomiphene citrate', 'Clotrimazole', 'Cyclopiazonic acid',
    'Desipramine hydrochloride', 'Dyclonine hydrochloride', 'EDTA', 'Emodin', 'FCCP',
    'Geldanamycin', 'Haloperidol', 'Harmine hydrochloride', 'Hoechst 33258', 'Hydroxyurea',
    'LY 294002 hydrochloride', 'NaCl', 'Nigericin', 'Nocodazole', 'Nystatin dihydrate',
    'Oligomycin', 'Parthenolide', 'Pentamidine isethionate', 'Raloxifene hydrochloride',
    'Rapamycin', 'SDS', 'Sorbitol', 'Staurosporine', 'Sulfometuron methyl', 'Trichostatin A',
    'Trifluoperazine dihydrochloride', 'Tunicamycin', 'U-73122', 'Valinomycin', 'Wortmannin',
]

def pubchem_smiles(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(name)}/property/IsomericSMILES/JSON"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    r = urllib.request.urlopen(req, timeout=30)
    data = json.loads(r.read().decode('utf-8'))
    props = data['PropertyTable']['Properties'][0]
    # 兼容多种 SMILES 字段名
    for k in ('SMILES', 'IsomericSMILES', 'CanonicalSMILES', 'ConnectivitySMILES'):
        if k in props and props[k]:
            return props[k], props.get('CID')
    return None, props.get('CID')

result = {}
failed = []
for c in COMPOUNDS:
    qname = NAME_MAP.get(c, c)
    done = False
    for name in [qname, c]:
        try:
            smi, cid = pubchem_smiles(name)
            result[c] = {'query_name': name, 'cid': cid, 'smiles': smi}
            print(f"[OK] {c} -> {smi[:55] if smi else 'NO_SMILES'}")
            done = True
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue  # 换原名重试
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
        print(f"[FAIL] {c}: 所有名称都查不到")
    time.sleep(0.3)

print(f"\n成功 {len(result)}/{len(COMPOUNDS)}，失败 {len(failed)}")
if failed:
    print('失败:', failed)

with open('data/chem_smiles.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print('已保存 data/chem_smiles.json')
