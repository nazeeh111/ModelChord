"""Finalize pre-confirmation selection ledger and measure composer peak memory."""
from pathlib import Path
import sys,json,resource,time,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_study as s
import torch,numpy as np
from modelchord import Artifact,merge,repair,Capability

root=s.ROOT
registry=s.load(root/'study/recipe-registry.json')
for row in (s.load(root/'study/inversion-training-diagnostic.json') if (root/'study/inversion-training-diagnostic.json').exists() else []):
 if not any(r.get('artifact')==row['artifact'] and r.get('name')=='inversion-training-diagnostic' for r in registry):registry.append({'suite':'vision','name':'inversion-training-diagnostic','data_assisted':True,**row})
s.write(root/'study/recipe-registry.json',registry)
vision=s.load(root/'study/vision-pending.json');text=s.load(root/'study/text-pending.json');selected={}
# Bind the processed-data fingerprint as a metadata-only ledger addition.
texts,labels,_,_=s.sms_data();ledger=s.load(root/'study/text-split.json')
if 'processed_sha256' not in ledger:
 original=root/'study/pilot-v1/text-split.registration.json';original.parent.mkdir(parents=True,exist_ok=True)
 if not original.exists():original.write_bytes((root/'study/text-split.json').read_bytes())
 ledger['processed_sha256']=s.hashlib.sha256(s.canonical({'texts':texts,'labels':labels.tolist()})).hexdigest();s.write(root/'study/text-split.json',ledger)
 text['original_split_sha256']=text['split_sha256'];text['current_split_ledger_sha256']=s.digest(root/'study/text-split.json');s.write(root/'study/text-pending.json',text)
# Same-start joint comparator for the strongest-method repair, not just TA repair.
npz=np.load(root/'study/workbench.npz',allow_pickle=False)
cal={t:(torch.tensor(npz[f'cal_x_{t}']),torch.tensor(npz[f'cal_y_{t}'])) for t in s.TASKS};sel={t:(torch.tensor(npz[f'x_{t}']),torch.tensor(npz[f'y_{t}'])) for t in s.TASKS}
pool_x=torch.cat([cal[t][0] for t in s.TASKS]);pool_y=torch.cat([cal[t][1] for t in s.TASKS])
for seed in vision['seeds']:
 strongest=next((r for r in seed['rows'] if r['name']=='repair-strongest'),None)
 if strongest is None or any(r['name']=='joint_finetune_strongest' for r in seed['rows']):continue
 ordinary=next(r for r in seed['rows'] if r['name']==strongest['ordinary_method']);a=Artifact.open(root/ordinary['artifact'])
 report=s.load(root/f"study/vision-seed{seed['seed']}-strongest-repair.json");targets={entry['round']:entry['target'] for entry in report['trace']};candidates=[]
 for lr in [.002,.006,.02]:
  m=a.model();opt=torch.optim.SGD(m.parameters(),lr=lr);gen=torch.Generator().manual_seed(8801);consumed=0;started=time.perf_counter()
  for _,target in sorted(targets.items()):
   count=len(cal[target][1]);ids=torch.randperm(len(pool_y),generator=gen)[:count];opt.zero_grad();s.F.cross_entropy(m(pool_x[ids]),pool_y[ids]).backward();opt.step();consumed+=count
  assert consumed==report['gradient_examples']
  out=root/strongest['artifact'];out=out.parent/f'joint-strongest-{lr}'
  art=s.export(out,m,a.manifest['contract'],a.manifest['base_id'],operation='same-start ordinary joint mixture finetuning',ordinary_weights_sha256=a.manifest['weights_sha256'],lr=lr,gradient_examples=consumed,gradient_steps=len(targets),teacher_calls=0)
  metrics=s.score(m,sel);row={'name':'joint_finetune_strongest','ordinary_method':strongest['ordinary_method'],'artifact':str(art.path.relative_to(root)),'lr':lr,'gradient_examples':consumed,'gradient_steps':len(targets),'teacher_calls':0,'selection':metrics,'selection_objective':list(s.objective(metrics)),'build_seconds':time.perf_counter()-started,'selection_examples':sum(len(b[1]) for b in sel.values()),'matching':'same starting ordinary weights as strongest repair; equal total gradient example accesses/steps/teacher calls; distinct selection cost'}
  candidates.append(row);registry.append({'suite':'vision','seed':seed['seed'],**row})
 seed['rows'].append(min(candidates,key=lambda r:tuple(r['selection_objective'])))
 base_row=next(r for r in seed['rows'] if r['name']=='base');base_clean=s.classification(Artifact.open(root/base_row['artifact']).model(),*sel['clean'])['accuracy']
 seed['base_clean_selection_accuracy']=base_clean;seed['acceptable_clean_base_regression']=.10
 assert base_clean-.10 <= s.FLOORS['clean']+1e-9, 'Fixed clean floor no longer implies base regression bound'
 for row in seed['rows']:
  if 'selection' in row:row['selection_clean_base_regression']=base_clean-row['selection']['clean']['accuracy']
  row['distillation_teacher_calls']=0
  row['expert_activation_forward_calls']=3 if row['name']=='regmean' else 0
  row['expert_activation_examples']=3*len(cal['clean'][1]) if row['name']=='regmean' else 0
s.write(root/'study/vision-pending.json',vision);s.write(root/'study/recipe-registry.json',registry)

for name,pending,floors in [('vision',vision['seeds'][0],s.FLOORS),('text',text,text['floors'])]:
 eligible=[r for r in pending['rows'] if r['name'] in ['task_arithmetic','ties','dare','tsv','regmean','repair','repair-strongest']]
 for r in eligible:
  if 'selection' not in r:raise RuntimeError('Missing measured selection metrics')
 winner=min(eligible,key=lambda r:s.objective(r['selection'],floors))
 selected[name]={'artifact':winner['artifact'],'method':winner['name'],'selection':winner['selection'],'floors':floors,'selection_objective':list(s.objective(winner['selection'],floors)),'acceptable_on_selection':all(winner['selection'][t]['accuracy']+1e-9>=f for t,f in floors.items()),'status':'acceptable candidate found' if all(winner['selection'][t]['accuracy']+1e-9>=f for t,f in floors.items()) else 'no acceptable candidate found','selection_rule':'lowest worst deficit, then sum deficits, then mean cross entropy; main composed artifacts only, excluding single experts, joint baselines and controls'}
s.write(root/'study/deployment-selection.json',selected)
# Repeat the fixed ordinary+repair build in a fresh output directory; does not tune.
seed=vision['seeds'][0];ordinary_row=next(r for r in seed['rows'] if r['name']=='task_arithmetic')
base=Artifact.open(root/next(r['artifact'] for r in seed['rows'] if r['name']=='base'));experts=[Artifact.open(root/p) for p in seed['experts']]
npz=np.load(root/'study/workbench.npz',allow_pickle=False)
cal={t:(torch.tensor(npz[f'cal_x_{t}']),torch.tensor(npz[f'cal_y_{t}'])) for t in s.TASKS};sel={t:(torch.tensor(npz[f'x_{t}']),torch.tensor(npz[f'y_{t}'])) for t in s.TASKS}
start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='modelchord-measure-') as td:
 a=merge(base,experts,Path(td)/'ordinary',method='task_arithmetic',scale=ordinary_row['parameters']['scale'],density=ordinary_row['parameters']['density'],seed=20261029)
 b,report=repair(a,cal,sel,s.CAPS,Path(td)/'repaired',budget=s.BUDGET)
 expected=Artifact.open(root/next(r['artifact'] for r in seed['rows'] if r['name']=='repair'))
 measured={'scope':'fresh finalization process including data-ledger checks, same-start joint controls, conventional merge and repair; excluding expert training','peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'wall_seconds':time.perf_counter()-start,'weights_reproduce_exactly':b.manifest['weights_sha256']==expected.manifest['weights_sha256'],'rank_used':report['rank_used_upper_bound'],'gradient_examples':report['gradient_examples'],'examples_evaluated_total':report['examples_evaluated_total'],'teacher_calls':0}
 if not measured['weights_reproduce_exactly']:raise RuntimeError('Fixed build did not reproduce weights')
 s.write(root/'study/build-resource-measurement.json',measured)
from data_registry import normalize
normalized=normalize(root)
print(json.dumps({'deployment_selection':selected,'build_measurement':measured,'bound_registry_entries':len(normalized)},indent=2),flush=True)
