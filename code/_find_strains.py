# -*- coding: utf-8 -*-
"""查 6 个菌株的 NCBI assembly accession"""
import urllib.request, urllib.parse, json, time

def esearch(term):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=assembly&term={urllib.parse.quote(term)}&retmode=json&retmax=3"
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    r = urllib.request.urlopen(req, timeout=30)
    d = json.loads(r.read().decode('utf-8'))
    return d.get('esearchresult', {}).get('idlist', [])

strains = ['BAH', 'BAI', 'CEK', 'CGD', 'CRD', 'DHY210']
# 尝试多种查询模式
for s in strains:
    found = None
    for pattern in [f'{s}.nuclear_genome', f'{s} [Assembly Name]', f'Saccharomyces cerevisiae {s}']:
        try:
            ids = esearch(pattern)
            if ids:
                found = (pattern, ids)
                break
        except Exception as e:
            pass
        time.sleep(0.3)
    print(f'{s}: {found}')
