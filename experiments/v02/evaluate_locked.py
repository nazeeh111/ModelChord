"""v0.2 freeze + one-shot new-source confirmation. No fitting in this module."""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time,resource
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples'),str(HERE)]
import numpy as np,torch
from modelchord import Artifact
from modelchord.artifacts import canonical,digest
from text_features import text_features,transform_text
from repair_search import objective,classification

def write(path,x):path.write_bytes(canonical(x)+b'\n')
def load(path):return json.loads(Path(path).read_text())
def historical_v1():
 lock=load(ROOT/'study/confirmation-lock.json');bad=[];exceptions={}
 for group in ['files','artifact_weights']:
  for name,sha in lock[group].items():
   path=ROOT/name
   if name=='src/modelchord/__init__.py':
    old=HERE/'v01-source-snapshot'/name
    if digest(old)!=sha:bad.append(name+' original snapshot')
    if path.read_bytes()!=old.read_bytes().replace(b'__version__ = "0.1.0"',b'__version__ = "0.2.0"'):bad.append(name+' exceeds declared version-only exception')
    if digest(path)!=sha:exceptions[name]={'reason':'version metadata only, 0.1.0 to0.2.0','original_sha256':sha,'current_sha256':digest(path),'original_path':str(old.relative_to(ROOT))}
   elif digest(path)!=sha:bad.append(name)
 if bad:raise RuntimeError(f'v0.1 artifacts/recipes/numerics changed: {bad}')
 return {'verified_original_bound_paths':sum(len(lock[g]) for g in ['files','artifact_weights']),'metadata_exceptions':exceptions,'all_original_bytes_retained':True}

def participants():
 registry=load(HERE/'main/registry.json');out=[]
 expected={('vision',11,m) for m in ['repair','joint','fixed_factor','no_priority','fixed_rank','no_protection']}|{('vision',29,m) for m in ['repair','joint','fixed_factor']}|{('text',43,m) for m in ['repair','joint','fixed_factor']}
 if {(e['suite'],e['seed'],e['method']) for e in registry}!=expected or len(registry)!=len(expected):raise RuntimeError('Declared main development suite is incomplete or duplicated')
 for entry in registry:
  if digest(ROOT/entry['artifact']/'model.safetensors')!=entry['weights_sha256'] or digest(ROOT/entry['artifact']/'manifest.json')!=entry['manifest_sha256'] or digest((ROOT/entry['artifact']).parent/'report.json')!=entry['report_sha256']:
   raise RuntimeError('Completed development artifact/report identity changed')
  # Registration retains the source snapshot used for the completed trajectory.
  out.append({'suite':entry['suite'],'seed':entry['seed'],'method':entry['method'],'artifact':entry['artifact'],'selection_status':entry['status'],'role':'v2'})
 old=load(ROOT/'study/results.json')
 allowed={'base','expert-left','expert-right','expert-inverted','task_arithmetic','ties','dare','tsv','regmean'}
 for seed in old['vision']:
  for row in seed['rows']:
   if row['name'] in allowed:out.append({'suite':'vision','seed':seed['seed'],'method':row['name'],'artifact':row['artifact'],'role':'v1_fixed_baseline'})
 for row in old['text']['rows']:
  if row['name']!='repair':out.append({'suite':'text','seed':43,'method':row['name'],'artifact':row['artifact'],'role':'v1_fixed_baseline'})
 return out

def freeze():
 if (HERE/'confirmation-lock.json').exists():raise FileExistsError('Freeze is immutable')
 before=historical_v1();entries=participants()
 deployments={'vision':'experiments/v02/main/vision-seed11-repair/model','text':'experiments/v02/main/text-seed43-repair/model'}
 for suite,path in deployments.items():
  row=next(e for e in entries if e['artifact']==path)
  if row['selection_status']!='acceptable candidate found':raise RuntimeError('Primary selection gate failed')
 files=set()
 for entry in entries:
  folder=ROOT/entry['artifact'];files.update(folder.glob('*'))
 # Include all attempts/source snapshots and mechanical cohort/selection ledgers.
 for parent in [HERE/'main',HERE/'main-v1',HERE/'pilots',HERE/'data',HERE/'v01-source-snapshot']:
  files.update(p for p in parent.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 files.update(p for p in HERE.glob('*.py'));files.add(HERE/'PROTOCOL.md');files.add(HERE/'text-development.npz')
 files.update(ROOT/'src/modelchord'/name for name in ['artifacts.py','models.py','evaluate.py','repair_v2.py','__init__.py'])
 files.update(ROOT/name for name in ['examples/text_features.py','study/workbench.npz','study/vision-split.json','study/text-split.json','study/confirmation-lock.json','study/results.json','pyproject.toml','requirements-macos.lock'])
 if (ROOT/'src/modelchord/repair_v2.py').read_bytes()!=(HERE/'repair_search.py').read_bytes():raise RuntimeError('Promoted helper differs from studied helper')
 lock={'schema':'modelchord-v02-confirmation-lock','created_unix':time.time(),'new_confirmation_metrics_seen':False,'participants':entries,'deployments':deployments,'v01_integrity':before,
  'files':{str(p.relative_to(ROOT)):digest(p) for p in sorted(files)},'required_floors':{'vision':{'clean':.88,'left':.8,'right':.8,'inverted':.8},'text':{'clean':.75,'uppercase':.72,'leet':.72,'underscores':.72}},'planning_margin':.03,
  'text_reporting':'all eligible pooled and every video subgroup; shifted transfer, not fresh SMS confirmation','environment':environment()}
 write(HERE/'confirmation-lock.json',lock);print(json.dumps({'lock_sha256':digest(HERE/'confirmation-lock.json'),'bound_files':len(lock['files']),'participants':len(entries),'deployments':deployments},indent=2))

def verify():
 lock=load(HERE/'confirmation-lock.json');bad=[name for name,sha in lock['files'].items() if digest(ROOT/name)!=sha]
 if bad:raise RuntimeError('Confirmation lock mismatch '+str(bad))
 historical_v1();return lock

def environment():
 import platform,scipy,sklearn,safetensors
 return {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'safetensors':safetensors.__version__,'threads':2,'device':'cpu'}

def transformed(x,task):
 a=x.reshape(-1,8,8).clone()
 if task=='left':a[:,:,:-1]=a[:,:,1:].clone();a[:,:,-1]=0
 elif task=='right':a[:,:,1:]=a[:,:,:-1].clone();a[:,:,0]=0
 elif task=='inverted':a=1-a
 return a.reshape(-1,64)

def cohorts():
 manifest=load(HERE/'data/source-manifest.json')
 for name,info in manifest['files'].items():
  if name.startswith('raw/') and digest(HERE/'data'/name)!=info['sha256']:raise RuntimeError('Raw source hash mismatch '+name)
 index=load(HERE/'data/vision-collision-v2/vision-confirmation-index.json')
 a=np.loadtxt(HERE/'data/raw/optdigits.tra',delimiter=',',dtype=np.int64);ids=index['eligible_row_indices']
 x=torch.tensor(a[ids,:64]/16,dtype=torch.float32);y=torch.tensor(a[ids,64],dtype=torch.long)
 vision={n:(transformed(x,n),y) for n in ['clean','left','right','inverted']}
 ti=load(HERE/'data/text-transfer-index.json');sources={name:list(csv.DictReader((HERE/'data/raw'/name).open(encoding='utf-8-sig'))) for name in ti['raw_files']}
 texts=[];labels=[];videos=[]
 for row in ti['eligible_records']:
  raw=sources[row['source_file']][row['row_index_zero_based']]
  if hashlib.sha256(raw['CONTENT'].encode()).hexdigest()!=row['text_sha256_utf8'] or int(raw['CLASS'])!=row['label']:raise RuntimeError('Text row identity mismatch')
  texts.append(raw['CONTENT']);labels.append(row['label']);videos.append(row['source_file'])
 if any(len(t)>10000 for t in texts):raise RuntimeError('Text exceeds frozen preprocessing contract')
 ty=torch.tensor(labels,dtype=torch.long)
 text={n:(torch.tensor(text_features(transform_text(texts,n))),ty) for n in ['clean','uppercase','leet','underscores']}
 return vision,text,videos

def summary(metrics,floors):
 deficits={n:max(0.,floors[n]-m['accuracy']) for n,m in metrics.items()}
 return {'fraction_floors_met':sum(d<=1e-9 for d in deficits.values())/len(deficits),'worst_deficit':max(deficits.values()),'deficits':deficits}

def confirm():
 lock=verify();marker=HERE/'confirmation-started.json'
 if environment()!=lock['environment']:raise RuntimeError('Frozen evaluation environment changed')
 if marker.exists() or (HERE/'confirmation-results.json').exists():raise FileExistsError('New confirmation already started or observed; new tuning requires a new cohort')
 with marker.open('x') as f:f.write(canonical({'lock_sha256':digest(HERE/'confirmation-lock.json'),'started_unix':time.time()}).decode())
 torch.set_num_threads(2);started=time.perf_counter();vision,text,videos=cohorts();rows=[];parents={};bases={}
 for e in lock['participants']:
  if e['method'] in ['regmean' if e['suite']=='vision' else 'task_arithmetic']:parents[(e['suite'],e['seed'])]=e['artifact']
  if e['method']=='base':bases[(e['suite'],e['seed'])]=e['artifact']
 references={};accesses={'task_examples':0,'reference_clean_examples':0,'per_video_examples':0}
 for key,path in parents.items():
  data=vision if key[0]=='vision' else text
  parent_clean=classification(Artifact.open(ROOT/path).model(),*data['clean'])['accuracy'];base_clean=classification(Artifact.open(ROOT/bases[key]).model(),*data['clean'])['accuracy']
  references[key]=(parent_clean,base_clean);accesses['reference_clean_examples']+=2*len(data['clean'][1])
 for entry in lock['participants']:
  data=vision if entry['suite']=='vision' else text;model=Artifact.open(ROOT/entry['artifact']).model();metrics={n:classification(model,*batch) for n,batch in data.items()};floors=lock['required_floors'][entry['suite']]
  parent_clean,base_clean=references[(entry['suite'],entry['seed'])];accesses['task_examples']+=sum(len(y) for x,y in data.values())
  row={**entry,'metrics':metrics,'required':summary(metrics,floors),'planning':summary(metrics,{n:min(1.,v+.03) for n,v in floors.items()}),
   'protected_parent_clean_regression':parent_clean-metrics['clean']['accuracy'],'base_clean_regression':base_clean-metrics['clean']['accuracy'],'source_count':len(next(iter(data.values()))[1]),
   'artifact_bytes':sum(p.stat().st_size for p in (ROOT/entry['artifact']).iterdir()),'weights_bytes':(ROOT/entry['artifact']/'model.safetensors').stat().st_size}
  parent_margin=.02 if entry['suite']=='vision' else .03
  parent_pass=row['protected_parent_clean_regression']<=parent_margin+1e-12
  base_pass=row['base_clean_regression']<=.10+1e-12 if entry['suite']=='vision' else True
  selection_pass=entry.get('selection_status')=='acceptable candidate found'
  accepted=selection_pass and row['required']['fraction_floors_met']==1. and parent_pass and base_pass
  selected=entry['artifact']==lock['deployments'][entry['suite']]
  row['acceptance']={'planning_selection_passed':selection_pass,'required_confirmation_floors_passed':row['required']['fraction_floors_met']==1.,'protected_parent_bound':parent_margin,'protected_parent_passed':parent_pass,'vision_base_bound':.10 if entry['suite']=='vision' else None,'base_bound_passed':base_pass,'accepted_under_frozen_build_contract':accepted,'selected_for_deployment':selected,'deployment_accepted':selected and accepted,'cohort_scope':'writer-disjoint optical confirmation' if entry['suite']=='vision' else 'YouTube distribution-shifted transfer; not fresh SMS'}
  if entry['suite']=='text':
   row['per_video']={video:{n:classification(model,x[torch.tensor([v==video for v in videos])],y[torch.tensor([v==video for v in videos])]) for n,(x,y) in data.items()} for video in sorted(set(videos))}
   accesses['per_video_examples']+=sum(len(y) for x,y in data.values())
  rows.append(row);print(json.dumps({'suite':entry['suite'],'seed':entry['seed'],'method':entry['method'],'role':entry['role'],'accuracy':{n:round(m['accuracy'],5) for n,m in metrics.items()},'required':row['required']},sort_keys=True),flush=True)
 results={'schema':'modelchord-v02-results','lock_sha256':digest(HERE/'confirmation-lock.json'),'rows':rows,'wall_seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'evaluation_example_accesses':accesses,'text_scope':'distribution-shifted YouTube transfer only; no fresh SMS confirmation','deployment_selection_remains_fixed':lock['deployments']}
 write(HERE/'confirmation-results.json',results);verify();write(HERE/'post-confirmation-integrity.json',{'lock_sha256':digest(HERE/'confirmation-lock.json'),'results_sha256':digest(HERE/'confirmation-results.json'),'all_bound_files_unchanged':True,'bound_file_count':len(lock['files'])})

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','verify','confirm','audit-data']);args=p.parse_args()
 if args.phase=='freeze':freeze()
 elif args.phase=='verify':print(json.dumps({'verified_files':len(verify()['files'])}))
 elif args.phase=='audit-data':
  a,b,v=cohorts();print(json.dumps({'vision_sources':len(a['clean'][1]),'text_sources':len(b['clean'][1]),'videos':len(set(v)),'model_metrics':False}))
 else:confirm()
if __name__=='__main__':main()
