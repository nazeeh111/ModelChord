#!/usr/bin/env python3
"""Exercise real local server endpoints and retain evidence, without a browser."""
import json
import urllib.request
import urllib.error
from pathlib import Path
import zipfile
import io
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:8765'
def request(route,data=None,headers=None):
    body=json.dumps(data).encode() if data is not None else None
    req=urllib.request.Request(BASE+route,data=body,headers=headers or ({'Content-Type':'application/json'} if body else {}))
    with urllib.request.urlopen(req,timeout=120) as r:return json.loads(r.read())

catalog=request('/api/catalog');assert len(catalog['experts'])>=3 and all(e['compatible'] for e in catalog['experts'])
sample=request('/api/sample?domain=clean')
inputs=sample['inputs'];first=request('/api/predict',{'artifact':'base','inputs':inputs})
changed=[inputs[0].copy()];changed[0][::2]=[1-v for v in changed[0][::2]]
second=request('/api/predict',{'artifact':'base','inputs':changed});assert first['probabilities']!=second['probabilities']
result=request('/api/build',{'experts':[e['id'] for e in catalog['experts']],'method':'regmean','scale':.5,'repair':True,'budget':{'rank':8,'rounds':4},'floors':{c['name']:c['floor'] for c in catalog['capabilities']}})
with urllib.request.urlopen(BASE+result['download_url']) as response:export=response.read()
with zipfile.ZipFile(io.BytesIO(export)) as z:assert {'manifest.json','model.safetensors','build-report.json'}<=set(z.namelist())
checks={}
for name,route,data,headers in [
 ('reject traversal','/../pyproject.toml',None,None),
 ('reject unknown expert','/api/build',{'experts':['../../private'],'floors':{}},None),
 ('reject cross-origin','/api/predict',{'artifact':'base','inputs':inputs},{'Content-Type':'application/json','Origin':'http://untrusted.example'}),
 ('reject nonfinite','/api/predict',{'artifact':'base','inputs':[[float('nan')]*64]},None),
 ('reject unknown artifact','/api/predict',{'artifact':'../../private','inputs':inputs},None)]:
    try:request(route,data,headers);checks[name]=False
    except urllib.error.HTTPError as exc:checks[name]=exc.code in {400,403,404}
assert all(checks.values()),checks
out={'catalog_experts':len(catalog['experts']),'changed_input_changed_output':True,'build_id':result['id'],
     'repair_status':result['report']['status'],'metrics':result['metrics'],'zip_bytes':len(export),'failure_checks':checks}
(ROOT/'study/workbench-checks-final.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
