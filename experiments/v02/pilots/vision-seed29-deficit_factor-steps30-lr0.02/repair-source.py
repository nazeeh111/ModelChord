"""v0.2 development: bounded trained factors and search-matched controls.

No v0.1 numerical source is edited. Selection gates accept whole trajectories,
which may cross temporary bad states; rejected trajectories are discarded.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import time
import math
import torch
from torch import nn
from torch.nn import functional as F
from modelchord import Artifact, export_artifact, Capability
from modelchord.evaluate import classification
from modelchord.artifacts import canonical

@dataclass(frozen=True)
class SearchBudget:
    rank: int = 8
    stages: int = 4
    steps_per_trial: int = 30
    factor_lr: float = .02
    max_block_norm: float = 4.
    temperature: float = .08
    seed: int = 20261030

class FactorLinear(nn.Module):
    def __init__(self, linear, rank, seed):
        super().__init__();self.base=deepcopy(linear)
        for p in self.base.parameters():p.requires_grad_(False)
        generator=torch.Generator().manual_seed(seed)
        raw=torch.randn(linear.in_features,rank,generator=generator)
        q,_=torch.linalg.qr(raw,mode='reduced')
        self.a=nn.Parameter(q.T.contiguous());self.b=nn.Parameter(torch.zeros(linear.out_features,rank))
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.a),self.b)
    def fold(self):
        out=deepcopy(self.base)
        with torch.no_grad():out.weight.add_(self.b@self.a)
        return out
    def project(self,bound):
        with torch.no_grad():
            norm=float((self.b@self.a).norm())
            if norm>bound:self.b.mul_(bound/norm)

def measure(model,data):return {name:classification(model,*batch) for name,batch in data.items()}
def objective(metrics,floors):
    deficits=[max(0.,floors[n]-metrics[n]['accuracy']) for n in floors]
    return max(deficits),sum(deficits),sum(metrics[n]['cross_entropy'] for n in floors)/len(floors)
def data_hashes(data):
    return {n:hashlib.sha256(b''.join(canonical({'dtype':str(x.dtype),'shape':list(x.shape)})+x.detach().cpu().contiguous().numpy().tobytes() for x in batch)).hexdigest() for n,batch in data.items()}

def validate_inputs(ordinary,calibration,selection,caps,budget):
    names=[c.name for c in caps]
    if not names or len(names)>16 or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):
        raise ValueError('Nonempty unique capability names required')
    if set(names)!=set(calibration) or set(names)!=set(selection):raise ValueError('Capability/data mapping mismatch')
    if any(not math.isfinite(c.floor) or not 0<=c.floor<=1 or not math.isfinite(c.max_regression) or not 0<=c.max_regression<=1 for c in caps):raise ValueError('Finite floors/margins in [0,1] required')
    if any(type(v) is not int for v in [budget.rank,budget.stages,budget.steps_per_trial,budget.seed]):raise ValueError('Integer budgets required')
    if not 1<=budget.stages<=8 or budget.stages*2>budget.rank or budget.rank>64 or not 1<=budget.steps_per_trial<=256:raise ValueError('Bounded stages/rank/steps required')
    if not math.isfinite(budget.factor_lr) or not 0<budget.factor_lr<=.2 or not math.isfinite(budget.max_block_norm) or not 0<budget.max_block_norm<=8 or not math.isfinite(budget.temperature) or not 0<budget.temperature<=1:raise ValueError('Invalid learning/correction budget')
    arch=ordinary.manifest['architecture']
    if len(arch['hidden'])!=2 or min([arch['input_dim'],*arch['hidden']])<3 or len(arch['classes'])<2:raise ValueError('v2 supports the declared three-layer classifier contract')
    for data in [calibration,selection]:
        for x,y in data.values():
            if x.ndim!=2 or x.shape[1]!=arch['input_dim'] or y.ndim!=1 or len(x)!=len(y) or not 1<=len(y)<=8192 or x.dtype!=torch.float32 or y.dtype!=torch.long or not torch.isfinite(x).all() or y.min()<0 or y.max()>=len(arch['classes']):raise ValueError('Invalid bounded finite classification batch')
    if set(data_hashes(calibration).values())&set(data_hashes(selection).values()):raise ValueError('Identical calibration/selection batches')
    return names

def search(ordinary,calibration,selection,caps,output,*,mode='deficit_factor',budget=SearchBudget()):
    if mode not in {'deficit_factor','uniform_factor','joint'}:raise ValueError('Unknown search mode')
    if budget.stages*2>budget.rank or not 1<=budget.stages<=8 or not 1<=budget.steps_per_trial<=256:raise ValueError('Bounded stages/rank/steps required')
    if not 0<budget.factor_lr<=.2 or not 0<budget.max_block_norm<=8:raise ValueError('Invalid step or correction budget')
    names=validate_inputs(ordinary,calibration,selection,caps,budget)
    cal_hash=data_hashes(calibration);sel_hash=data_hashes(selection)
    if set(cal_hash.values())&set(sel_hash.values()):raise ValueError('Identical calibration/selection batches')
    selection_examples=0
    def counted_measure(model):
        nonlocal selection_examples
        result={}
        for name,batch in selection.items():
            selection_examples+=len(batch[1]);result[name]=classification(model,*batch)
        return result
    current=ordinary.model();initial=counted_measure(current)
    floors={c.name:max(c.floor,initial[c.name]['accuracy']-c.max_regression) if c.protected else c.floor for c in caps}
    def gate(metrics):return all(not c.protected or metrics[c.name]['accuracy']+1e-9>=initial[c.name]['accuracy']-c.max_regression for c in caps)
    metrics=initial;rank_used=0;trace=[];gradient_examples=0;calibration_metric_examples=0;start=time.perf_counter()
    xx=torch.cat([calibration[n][0] for n in names]);yy=torch.cat([calibration[n][1] for n in names]);sizes=[len(calibration[n][1]) for n in names]
    stage_steps=[];gradient_steps=0
    for stage in range(budget.stages):
        steps_before=gradient_steps
        if objective(metrics,floors)[0]<=1e-9:break
        if mode=='joint':
            specs=[{'lr':lr,'loss':loss} for lr in [.001,.003,.01] for loss in ['uniform','deficit']]
        else:
            # Six functional interventions. Rank is assigned to the winning layer.
            specs=[{'layer':layer,'rank':rank,'lr':budget.factor_lr,'loss':'deficit' if mode=='deficit_factor' else 'uniform'} for layer in range(3) for rank in [1,2]]
        candidates=[]
        for index,spec in enumerate(specs):
            model=deepcopy(current);factor=None
            if mode!='joint':
                for p in model.parameters():p.requires_grad_(False)
                factor=FactorLinear(model.layers[spec['layer']],spec['rank'],budget.seed+100*stage+index)
                model.layers[spec['layer']]=factor
            else:
                for p in model.parameters():p.requires_grad_(True)
            parameters=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.Adam(parameters,lr=spec['lr'])
            failed=None;completed_steps=0
            for step in range(budget.steps_per_trial):
                optimizer.zero_grad(set_to_none=True)
                logits=model(xx);gradient_examples+=len(yy)
                if not torch.isfinite(logits).all():failed='nonfinite logits';break
                losses=[];accuracies=[];offset=0
                for size in sizes:
                    z=logits[offset:offset+size];target=yy[offset:offset+size];offset+=size
                    losses.append(F.cross_entropy(z,target));accuracies.append(z.detach().argmax(1).eq(target).float().mean())
                if spec['loss']=='deficit':
                    deficits=torch.stack([torch.relu(torch.tensor(floors[n])-a) for n,a in zip(names,accuracies)])
                    weights=torch.softmax(deficits/budget.temperature,0).clamp_min(.1);weights=weights/weights.sum()
                else:weights=torch.full((len(names),),1/len(names))
                loss=(torch.stack(losses)*weights).sum()
                if not torch.isfinite(loss):failed='nonfinite loss';break
                loss.backward();optimizer.step();gradient_steps+=1;completed_steps+=1
                if factor is not None:factor.project(budget.max_block_norm)
            if factor is not None:model.layers[spec['layer']]=factor.fold()
            try:
                result=counted_measure(model)
                accepted=failed is None and gate(result) and objective(result,floors)<objective(metrics,floors)
            except (ValueError,RuntimeError) as exc:
                failed=str(exc);result=None;accepted=False
            row={'stage':stage,'trial':index,'spec':spec,'metrics':result,'protected_gate':gate(result) if result else False,'failure':failed,'selected':False,'rank_after':rank_used+spec.get('rank',0),'steps_requested':budget.steps_per_trial,'steps_completed':completed_steps}
            trace.append(row)
            if accepted:candidates.append((objective(result,floors),spec.get('rank',0),model,result,len(trace)-1))
        stage_steps.append(gradient_steps-steps_before)
        if not candidates:break
        _,rank,current,metrics,trace_id=min(candidates,key=lambda c:(c[0],c[1],c[4]));trace[trace_id]['selected']=True;rank_used+=rank
    final=counted_measure(current)
    report={'algorithm':'trained_intervention_search/v2','mode':mode,'budget':asdict(budget),'initial':initial,'final':final,'effective_floors':floors,'trace':trace,
            'rank_used_upper_bound':rank_used,'gradient_steps_total_search':gradient_steps,'gradient_examples_total_search':gradient_examples,
            'selection_examples_total_search':selection_examples,'calibration_metric_examples':calibration_metric_examples,'teacher_calls':0,'trial_count':len(trace),
            'wall_seconds':time.perf_counter()-start,'calibration_sha256':cal_hash,'selection_sha256':sel_hash,'stage_search_steps':stage_steps,
            'status':'acceptable candidate found' if objective(final,floors)[0]<=1e-9 and gate(final) else 'no acceptable candidate found',
            'worst_deficit':objective(final,floors)[0],'fraction_floors_met':sum(final[n]['accuracy']+1e-9>=floors[n] for n in names)/len(names)}
    artifact=export_artifact(output,current,ordinary.manifest['contract'],base_id=ordinary.manifest['base_id'],license=ordinary.manifest['license'],provenance={'operation':'v2 bounded intervention search','ordinary_weights_sha256':ordinary.manifest['weights_sha256'],'report':report})
    return artifact,report

def reference_search(ordinary,calibration,selection,caps,output,*,mode='joint',budget=SearchBudget(),stages=3):
    """Six ordinary trajectories, same update/trial/selection ceilings as repair.

    All permitted updates execute, even if an earlier checkpoint is feasible;
    selection may choose that earlier checkpoint. Calibration source access is
    identical to factor search: every domain participates in every update.
    """
    if mode not in {'joint','fixed_factor'}:raise ValueError('Unknown reference')
    if not 1<=stages<=budget.stages:raise ValueError('Invalid matched stage count')
    names=validate_inputs(ordinary,calibration,selection,caps,budget);initial_model=ordinary.model();selection_examples=0
    def counted_measure(model):
        nonlocal selection_examples
        result={}
        for name,batch in selection.items():
            selection_examples+=len(batch[1]);result[name]=classification(model,*batch)
        return result
    initial=counted_measure(initial_model)
    floors={c.name:max(c.floor,initial[c.name]['accuracy']-c.max_regression) if c.protected else c.floor for c in caps}
    def gate(m):return all(not c.protected or m[c.name]['accuracy']+1e-9>=initial[c.name]['accuracy']-c.max_regression for c in caps)
    xx=torch.cat([calibration[n][0] for n in names]);yy=torch.cat([calibration[n][1] for n in names]);sizes=[len(calibration[n][1]) for n in names]
    if mode=='joint':specs=[{'lr':lr,'loss':loss} for lr in [.001,.006,.02] for loss in ['uniform','deficit']]
    else:specs=[{'lr':lr,'loss':'uniform'} for lr in [.001,.003,.006,.01,.02,.04]]
    best=initial_model;metrics=initial;trace=[];gradient_examples=0;gradient_steps=0;selected_rank=0;started=time.perf_counter()
    for trial,spec in enumerate(specs):
        model=ordinary.model();factors=[]
        if mode=='fixed_factor':
            for p in model.parameters():p.requires_grad_(False)
            for layer,rank in enumerate([3,3,2]):
                f=FactorLinear(model.layers[layer],rank,budget.seed+layer);model.layers[layer]=f;factors.append((layer,f))
        else:
            for p in model.parameters():p.requires_grad_(True)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=spec['lr']);failed=None
        for stage in range(stages):
            for step in range(budget.steps_per_trial):
                optimizer.zero_grad(set_to_none=True);logits=model(xx);gradient_examples+=len(yy)
                if not torch.isfinite(logits).all():failed='nonfinite logits';break
                losses=[];accs=[];offset=0
                for size in sizes:
                    z=logits[offset:offset+size];target=yy[offset:offset+size];offset+=size
                    losses.append(F.cross_entropy(z,target));accs.append(z.detach().argmax(1).eq(target).float().mean())
                if spec['loss']=='deficit':
                    deficits=torch.stack([torch.relu(torch.tensor(floors[n])-a) for n,a in zip(names,accs)])
                    weights=torch.softmax(deficits/budget.temperature,0).clamp_min(.1);weights/=weights.sum()
                else:weights=torch.full((len(names),),1/len(names))
                loss=(torch.stack(losses)*weights).sum()
                if not torch.isfinite(loss):failed='nonfinite loss';break
                loss.backward();optimizer.step();gradient_steps+=1
                for _,f in factors:f.project(budget.max_block_norm)
            folded=deepcopy(model)
            for layer,f in factors:folded.layers[layer]=f.fold()
            try:
                result=counted_measure(folded)
                accepted=failed is None and gate(result) and objective(result,floors)<objective(metrics,floors)
            except (ValueError,RuntimeError) as exc:failed=str(exc);result=None;accepted=False
            entry={'trial':trial,'stage':stage,'spec':spec,'metrics':result,'protected_gate':gate(result) if result else False,'failure':failed,'selected':False}
            trace.append(entry)
            if accepted:
                for t in trace:t['selected']=False
                trace[-1]['selected']=True;best=folded;metrics=result;selected_rank=8 if mode=='fixed_factor' else None
            if failed:break
    final=counted_measure(best)
    report={'algorithm':'matched_trajectory_reference/v2','mode':mode,'budget':asdict(budget),'matched_stages':stages,'initial':initial,'final':final,'effective_floors':floors,'trace':trace,
            'rank_used_upper_bound':selected_rank,'gradient_steps_total_search':gradient_steps,'gradient_examples_total_search':gradient_examples,'selection_examples_total_search':selection_examples,
            'teacher_calls':0,'trial_count':len(trace),'wall_seconds':time.perf_counter()-started,'calibration_sha256':data_hashes(calibration),'selection_sha256':data_hashes(selection),
            'status':'acceptable candidate found' if objective(final,floors)[0]<=1e-9 and gate(final) else 'no acceptable candidate found','worst_deficit':objective(final,floors)[0],
            'fraction_floors_met':sum(final[n]['accuracy']+1e-9>=floors[n] for n in names)/len(names)}
    artifact=export_artifact(output,best,ordinary.manifest['contract'],base_id=ordinary.manifest['base_id'],license=ordinary.manifest['license'],provenance={'operation':'v2 matched trajectory reference','ordinary_weights_sha256':ordinary.manifest['weights_sha256'],'report':report})
    return artifact,report
