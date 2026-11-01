#!/usr/bin/env python3
"""Fixed-recipe resource repeats; no new fitting policy or cohort access."""
from pathlib import Path
import argparse,json,sys,subprocess,tempfile,time,resource
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]

def child(method):
 import numpy as np,torch
 from modelchord import Artifact,Capability
 from modelchord.repair_v2 import SearchBudget,search,reference_search
 torch.set_num_threads(2);started=time.perf_counter()
 with np.load(ROOT/'study/workbench.npz',allow_pickle=False) as d:
  names=['clean','left','right','inverted'];cal={n:(torch.tensor(d[f'cal_x_{n}']),torch.tensor(d[f'cal_y_{n}'])) for n in names};sel={n:(torch.tensor(d[f'x_{n}']),torch.tensor(d[f'y_{n}'])) for n in names}
 parent=Artifact.open(ROOT/'artifacts/vision-seed11-v2/regmean-0');caps=[Capability(n,.91 if n=='clean' else .83,n=='clean',.02) for n in names]
 expected=Artifact.open(ROOT/f'experiments/v02/main/vision-seed11-{method}/model')
 budget=SearchBudget()
 with tempfile.TemporaryDirectory(prefix='modelchord-v02-build-profile-') as td:
  if method=='repair':a,r=search(parent,cal,sel,caps,Path(td)/'model',budget=budget)
  else:a,r=reference_search(parent,cal,sel,caps,Path(td)/'model',mode=method,budget=budget,stages=3)
  matched=a.manifest['weights_sha256']==expected.manifest['weights_sha256']
  if not matched:raise RuntimeError('Fixed repeat changed model weights')
  row={'method':method,'whole_build_seconds':time.perf_counter()-started,'helper_seconds':r['wall_seconds'],'fresh_process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'weights_match_frozen_candidate':matched,'gradient_accesses':r['gradient_examples_total_search'],'selection_accesses':r['selection_examples_total_search'],'max_simultaneous_trainable_parameters':max(t['trainable_parameters'] for t in r['trace']),
       'scope':'CPU2threads; includes data/parent load, selection, complete search and static export; process RSS includes imports; excludes expert training and inherited RegMean construction'}
 print(json.dumps(row),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--child',choices=['repair','joint','fixed_factor']);a=p.parse_args()
 if a.child:child(a.child);return
 rows=[]
 for method in ['repair','joint','fixed_factor']:
  started=time.perf_counter();r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',method],cwd=ROOT,capture_output=True,text=True,check=True)
  row=json.loads(r.stdout);row['cold_process_including_imports_seconds']=time.perf_counter()-started;rows.append(row);print(json.dumps(row),flush=True)
 (ROOT/'experiments/v02/build-profile.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
