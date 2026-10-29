"""Bind every attempted recipe, including retained pilots and source specialists."""
from pathlib import Path
import json,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from modelchord.artifacts import canonical,digest

def normalize(root):
 entries=json.loads((root/'study/recipe-registry.json').read_text());phase='pilot-v1';normalized=[]
 for number,entry in enumerate(entries):
  entry=dict(entry);artifact=entry.get('artifact')
  if artifact and 'vision-seed' in artifact and '-v2/' in artifact:phase='main-v2'
  if not artifact:
   if entry.get('suite')=='text':artifact='artifacts/text/repaired' if entry['name']=='repair' else None
   else:
    directory=f"artifacts/vision-seed{entry.get('seed',11)}"+('-v2' if phase=='main-v2' else '')
    name={'repair-strongest':'repair-strongest','easy-merge':'easy-repaired'}.get(entry['name'],entry['name'])
    artifact=f'{directory}/{name}'
  if not artifact or not (root/artifact/'manifest.json').exists():raise RuntimeError(f'Missing artifact binding for {entry}')
  entry['artifact']=artifact
  normalized.append(entry)
 seen={entry['artifact'] for entry in normalized}
 for manifest in sorted((root/'artifacts').rglob('manifest.json')):
  artifact=str(manifest.parent.relative_to(root))
  if artifact not in seen:
   m=json.loads(manifest.read_text());normalized.append({'suite':m['contract']['family'],'name':manifest.parent.name,'artifact':artifact,'source_artifact':True,'provenance':m['provenance']});seen.add(artifact)
 outputs=[]
 for number,entry in enumerate(normalized):
  artifact=entry['artifact'];path=root/artifact;manifest=json.loads((path/'manifest.json').read_text());report=manifest['provenance'].get('report')
  phase='text-v1' if entry.get('suite')=='text' else ('diagnostic-v1' if 'diagnostic' in artifact else ('main-v2' if '-v2/' in artifact else 'pilot-v1'))
  original=dict(entry)
  entry.update({'entry_id':f"{phase}:{entry.get('suite','vision')}:{entry.get('seed','na')}:{entry['name']}:{number:04d}",'phase':phase,'protocol_version':'v1+preconfirmation-amendments','protocol_sha256':digest(root/'docs/PROTOCOL.md'),'manifest_sha256':digest(path/'manifest.json'),'weights_sha256':digest(path/'model.safetensors'),'report_sha256':hashlib.sha256(canonical(report if report is not None else original)).hexdigest(),'report_location':'artifact manifest provenance.report' if report is not None else 'recipe entry before binding metadata','data_assisted':entry.get('data_assisted',manifest['provenance'].get('data_assisted',report is not None or bool(manifest['provenance'].get('training')) or bool(manifest['provenance'].get('adapter')) or bool(entry.get('gradient_examples')))),'distillation_teacher_calls':0})
  if entry['name']=='regmean':
   count=manifest['provenance'].get('calibration_examples',537)
   entry.update({'expert_activation_forward_calls':3,'expert_activation_examples':count,'total_evaluated_example_accesses':entry.get('selection_examples',0)+count,'teacher_calls_note':'No distillation targets; RegMean does run parent expert activation forwards, counted separately.'})
  elif report is not None:
   entry.update({'expert_activation_forward_calls':0,'expert_activation_examples':0,'total_evaluated_example_accesses':report.get('examples_evaluated_total',report['validation_examples']+report['gradient_examples']),'rank_used':report['rank_used_upper_bound']})
  else:entry.update({'expert_activation_forward_calls':0,'expert_activation_examples':0,'total_evaluated_example_accesses':entry.get('selection_examples',0)})
  provenance=manifest['provenance']
  training=provenance.get('training') or provenance.get('adapter')
  if training or (path.name=='expert-conflict'):
   suite=manifest['contract']['family'];n=988 if suite=='vision' else 718
   epochs=training.get('epochs',0) if training else 65
   entry.update({'stage':'expert_training','data_assisted':True,'training_example_accesses':epochs*n,'training_epochs':epochs,'training_sources_per_epoch':n,'composition_example_accesses':0,'training_access_note':'Repeated labeled training examples, reported separately from composition/calibration/selection accesses; per-checkpoint count is cumulative, not additive across diagnostic checkpoints.'})
  elif report is not None:
   entry.update({'stage':'calibration_repair','training_example_accesses':0,'composition_example_accesses':entry['total_evaluated_example_accesses']})
  elif entry.get('gradient_examples',0):
   entry.update({'stage':'calibration_finetuning_baseline','data_assisted':True,'training_example_accesses':0,'composition_example_accesses':entry.get('gradient_examples',0)+entry.get('selection_examples',0)})
  else:
   entry.update({'stage':'composition','training_example_accesses':0,'composition_example_accesses':entry['total_evaluated_example_accesses']})
  payload=report if report is not None else (entry.get('selection') if entry.get('selection') is not None else provenance)
  entry['report_sha256']=hashlib.sha256(canonical(payload)).hexdigest()
  entry['report_location']='artifact manifest provenance.report' if report is not None else ('recipe selection' if entry.get('selection') is not None else 'artifact manifest provenance')
  outputs.append(entry)
 (root/'study/recipe-registry.json').write_bytes(canonical(outputs)+b'\n')
 return outputs
