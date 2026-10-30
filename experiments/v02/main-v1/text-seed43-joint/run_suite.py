"""Execute the declared v0.2 development suite, never confirmation inference."""
from pathlib import Path
import sys,json,hashlib,resource,time
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'src'),str(HERE)]
import numpy as np,torch
from modelchord import Artifact,Capability
from modelchord.artifacts import canonical,digest
from repair_search import search,reference_search,SearchBudget

def write(path,x):path.write_bytes(canonical(x)+b'\n')
def main():
 torch.set_num_threads(2);base=HERE/'main';base.mkdir(exist_ok=True)
 protocol=HERE/'PROTOCOL.md';budget=SearchBudget();registry=[]
 for suite,seed in [('vision',11),('vision',29),('text',43)]:
  data_path=ROOT/('study/workbench.npz' if suite=='vision' else 'experiments/v02/text-development.npz')
  with np.load(data_path,allow_pickle=False) as d:
   names=['clean','left','right','inverted'] if suite=='vision' else ['clean','uppercase','leet','underscores']
   cal={n:(torch.tensor(d[f'cal_x_{n}']),torch.tensor(d[f'cal_y_{n}'])) for n in names};sel={n:(torch.tensor(d[f'x_{n}']),torch.tensor(d[f'y_{n}'])) for n in names}
  source=ROOT/(f'artifacts/vision-seed{seed}-v2/regmean-0' if suite=='vision' else 'artifacts/text/ordinary-3');ordinary=Artifact.open(source)
  required={n:(.88 if n=='clean' else .80) if suite=='vision' else (.75 if n=='clean' else .72) for n in names}
  caps=[Capability(n,required[n]+.03,n=='clean',.02 if suite=='vision' else .03) for n in names]
  for method in ['repair','joint','fixed_factor',*(['no_priority','fixed_rank','no_protection'] if suite=='vision' and seed==11 else [])]:
   folder=base/f'{suite}-seed{seed}-{method}';folder.mkdir()
   registration={'suite':suite,'seed':seed,'method':method,'budget':asdict(budget),'planning_margin':.03,'required_floors':required,'protocol_sha256':digest(protocol),
     'ordinary_weights_sha256':ordinary.manifest['weights_sha256'],'ordinary_manifest_sha256':digest(source/'manifest.json'),'development_data_sha256':digest(data_path),
     'source_sha256':{p.name:digest(p) for p in [HERE/'repair_search.py',Path(__file__)]},'confirmation_accessed':False}
   for p in [HERE/'repair_search.py',Path(__file__)]: (folder/p.name).write_bytes(p.read_bytes())
   if method in ['joint','fixed_factor']:
    preceding=json.loads((base/f'{suite}-seed{seed}-repair/report.json').read_text());stages=len(preceding['stage_search_steps']);registration['matched_repair_stages']=stages
   write(folder/'registered.json',registration)
   if method in ['joint','fixed_factor']:
    artifact,report=reference_search(ordinary,cal,sel,caps,folder/'model',mode=method,budget=budget,stages=stages)
    assert report['gradient_examples_total_search']==preceding['gradient_examples_total_search']
    assert report['selection_examples_total_search']==preceding['selection_examples_total_search']
    assert report['gradient_steps_total_search']==preceding['gradient_steps_total_search']
   else:
    artifact,report=search(ordinary,cal,sel,caps,folder/'model',mode='uniform_factor' if method=='no_priority' else 'deficit_factor',budget=budget,
      protect=method!='no_protection',fixed_rank=method=='fixed_rank')
   report['required_floors']=required;report['required_fraction_floors_met']=sum(report['final'][n]['accuracy']+1e-9>=required[n] for n in names)/len(names)
   report['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
   write(folder/'report.json',report)
   entry={**registration,'artifact':str(artifact.path.relative_to(ROOT)),'weights_sha256':artifact.manifest['weights_sha256'],'manifest_sha256':digest(artifact.path/'manifest.json'),'report_sha256':digest(folder/'report.json'),'status':report['status'],'rank':report['rank_used_upper_bound']}
   registry.append(entry);write(base/'registry.json',registry)
   print(json.dumps({'suite':suite,'seed':seed,'method':method,'status':report['status'],'rank':report['rank_used_upper_bound'],'accuracies':{n:v['accuracy'] for n,v in report['final'].items()},'gradient_accesses':report['gradient_examples_total_search'],'selection_accesses':report['selection_examples_total_search']},sort_keys=True),flush=True)
if __name__=='__main__':main()
