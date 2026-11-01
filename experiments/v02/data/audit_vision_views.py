"""Exact transformed-view collision audit; no model imports or inference.

Writes a complete new versioned directory atomically, preserving v1 ledgers.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib, json, os, shutil, tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parent
DEST = DATA / 'vision-collision-v2'
TASKS = ('clean', 'left', 'right', 'inverted')

def sha(b): return hashlib.sha256(b).hexdigest()
def digest(p): return sha(p.read_bytes())
def write(p,obj): p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def views(flat):
    x=np.asarray(flat,dtype=np.uint8).reshape(8,8)
    left=np.zeros_like(x);left[:,:-1]=x[:,1:]
    right=np.zeros_like(x);right[:,1:]=x[:,:-1]
    return dict(zip(TASKS,(x,left,right,16-x)))
def identity(a):
    assert a.dtype==np.uint8 and a.shape==(8,8)
    assert (a<=16).all()
    return sha(a.tobytes(order='C'))
def record(i,task,h,y): return {'source_row_index_zero_based':i,'task':task,'pixel_sha256_uint8_rowmajor':h,'label':int(y)}
def duplicate_groups(lookup,eligible=None):
    result=[]
    for h,rs in sorted(lookup.items()):
        chosen=[r for r in rs if eligible is None or r['source_row_index_zero_based'] in eligible]
        sources={r['source_row_index_zero_based'] for r in chosen}
        if len(sources)>1:
            result.append({'pixel_sha256_uint8_rowmajor':h,'distinct_source_count':len(sources),'distinct_labels':sorted({r['label'] for r in chosen}),'views':chosen})
    return result

if DEST.exists(): raise FileExistsError('Preserve existing audit version; choose a new version directory')
prior=json.loads((DATA/'vision-confirmation-index.json').read_text())
manifest=json.loads((DATA/'source-manifest.json').read_text())
for name in ('raw/optdigits.tra','raw/optdigits.tes'):
    assert digest(DATA/name)==manifest['files'][name]['sha256']
new=np.loadtxt(DATA/'raw/optdigits.tra',delimiter=',',dtype=np.uint8)
old=np.loadtxt(DATA/'raw/optdigits.tes',delimiter=',',dtype=np.uint8)
assert new.shape==(3823,65) and old.shape==(1797,65)
old_views=defaultdict(list);new_views=defaultdict(list);hash_records=[]
for i,row in enumerate(old):
    for task,a in views(row[:64]).items():
        h=identity(a);old_views[h].append(record(i,task,h,row[-1]))
for i,row in enumerate(new):
    hashes={}
    for task,a in views(row[:64]).items():
        h=identity(a);hashes[task]=h;new_views[h].append(record(i,task,h,row[-1]))
    hash_records.append({'source_row_index_zero_based':i,'label':int(row[-1]),'view_pixel_sha256_uint8_rowmajor':hashes})
collisions=[];dropped=set();pair_counts=Counter()
for h in sorted(set(old_views)&set(new_views)):
    olds=old_views[h];news=new_views[h]
    collisions.append({'pixel_sha256_uint8_rowmajor':h,'old_views':olds,'new_views':news})
    dropped.update(r['source_row_index_zero_based'] for r in news)
    for o in olds:
        for n in news: pair_counts[n['task']+'__'+o['task']]+=1
prior_eligible=set(prior['eligible_row_indices']);eligible=prior_eligible-dropped
for r in hash_records:
    i=r['source_row_index_zero_based'];r['eligible']=i in eligible
    if i in dropped:r['exclusion']='any_view_exact_pixel_match_to_any_view_of_all_1797_old_sources'
    elif i not in prior_eligible:r['exclusion']='prior_raw_pixel_filter'
allgroups=duplicate_groups(new_views);eligiblegroups=duplicate_groups(new_views,eligible)
audit={'schema_version':2,'audit':'every new source four views vs every old source four views','transformations':{'clean':'original integer pixels','left':'one-column left shift, final column zero','right':'one-column right shift, first column zero','inverted':'16 - original integer pixels'},'hash_representation':'SHA256 of 64 row-major uint8 counts in 0..16; labels and task names excluded','old_source_rows':len(old),'new_source_rows':len(new),'old_transformed_view_instances':len(old)*4,'new_transformed_view_instances':len(new)*4,'old_unique_transformed_view_hashes':len(old_views),'new_unique_transformed_view_hashes':len(new_views),'cross_source_collision_hash_count':len(collisions),'cross_source_collisions':collisions,'new_task__old_task_matching_view_pair_counts':dict(sorted(pair_counts.items())),'excluded_new_source_row_indices':sorted(dropped),'excluded_new_source_count':len(dropped),'eligible_new_source_count':len(eligible),'eligible_class_counts':dict(sorted(Counter(int(new[i,-1]) for i in eligible).items())),'within_new_distinct_source_collision_groups_all':allgroups,'within_new_distinct_source_collision_groups_eligible':eligiblegroups,'within_new_collision_summary':{'all_groups':len(allgroups),'eligible_groups':len(eligiblegroups),'all_source_rows_involved':len({r['source_row_index_zero_based'] for g in allgroups for r in g['views']}),'eligible_source_rows_involved':len({r['source_row_index_zero_based'] for g in eligiblegroups for r in g['views']}),'eligible_conflicting_label_groups':sum(len(g['distinct_labels'])>1 for g in eligiblegroups)},'model_inference_performed':False,'denominator_policy':'Exclude whole new source if ANY view matches ANY old view. Within-new collisions stay entirely within this confirmation partition; retain source-level denominator and report collisions, never treat 4N transformed instances as independent source samples.'}
index={'schema_version':2,'supersedes':'../vision-confirmation-index.json','dataset':prior['dataset'],'raw_file':'../raw/optdigits.tra','eligible_row_indices':sorted(eligible),'excluded_row_indices':sorted(set(range(len(new)))-eligible),'records':hash_records,'counts':{'eligible':len(eligible),'excluded':len(new)-len(eligible)},'audit_file':'transformed-view-audit.json','split_status':'whole eligible cohort confirmation only; no model metrics','model_inference_performed':False}
stage=Path(tempfile.mkdtemp(prefix='.vision-collision-v2-',dir=DATA))
try:
    write(stage/'vision-confirmation-index.json',index)
    write(stage/'transformed-view-audit.json',audit)
    provenance={'schema_version':2,'prepared_utc':'2026-10-29','model_inference_performed':False,'prior_ledger_sha256':digest(DATA/'vision-confirmation-index.json'),'prior_source_manifest_sha256':digest(DATA/'source-manifest.json'),'audit_script':'../audit_vision_views.py','audit_script_sha256':digest(Path(__file__)),'raw_tra_sha256':digest(DATA/'raw/optdigits.tra'),'raw_tes_sha256':digest(DATA/'raw/optdigits.tes'),'outputs':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(stage.iterdir())},'publication':'Only scripts, hashes and ledgers intended for publication; raw text remains local.'}
    write(stage/'manifest.json',provenance)
    os.rename(stage,DEST)
except BaseException:
    shutil.rmtree(stage);raise
print(json.dumps({k:audit[k] for k in ('old_transformed_view_instances','new_transformed_view_instances','cross_source_collision_hash_count','excluded_new_source_count','eligible_new_source_count','within_new_collision_summary')},indent=2))
