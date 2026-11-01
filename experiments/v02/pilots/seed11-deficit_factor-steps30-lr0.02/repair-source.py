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
    return {n:hashlib.sha256(b''.join(x.detach().cpu().contiguous().numpy().tobytes() for x in batch)).hexdigest() for n,batch in data.items()}

def search(ordinary,calibration,selection,caps,output,*,mode='deficit_factor',budget=SearchBudget()):
    if mode not in {'deficit_factor','uniform_factor','joint'}:raise ValueError('Unknown search mode')
    if budget.stages*2>budget.rank or not 1<=budget.stages<=8 or not 1<=budget.steps_per_trial<=256:raise ValueError('Bounded stages/rank/steps required')
    if not 0<budget.factor_lr<=.2 or not 0<budget.max_block_norm<=8:raise ValueError('Invalid step or correction budget')
    names=[c.name for c in caps]
    if set(names)!=set(calibration) or set(names)!=set(selection):raise ValueError('Capability/data mapping mismatch')
    cal_hash=data_hashes(calibration);sel_hash=data_hashes(selection)
    if set(cal_hash.values())&set(sel_hash.values()):raise ValueError('Identical calibration/selection batches')
    current=ordinary.model();initial=measure(current,selection)
    floors={c.name:max(c.floor,initial[c.name]['accuracy']-c.max_regression) if c.protected else c.floor for c in caps}
    def gate(metrics):return all(not c.protected or metrics[c.name]['accuracy']+1e-9>=initial[c.name]['accuracy']-c.max_regression for c in caps)
    metrics=initial;rank_used=0;trace=[];gradient_examples=0;selection_examples=sum(len(b[1]) for b in selection.values());calibration_metric_examples=0;start=time.perf_counter()
    xx=torch.cat([calibration[n][0] for n in names]);yy=torch.cat([calibration[n][1] for n in names]);sizes=[len(calibration[n][1]) for n in names]
    stage_steps=[];gradient_steps=0
    for stage in range(budget.stages):
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
            failed=None
            for step in range(budget.steps_per_trial):
                optimizer.zero_grad(set_to_none=True)
                logits=model(xx);gradient_examples+=len(yy);gradient_steps+=1
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
                loss.backward();optimizer.step()
                if factor is not None:factor.project(budget.max_block_norm)
            if factor is not None:model.layers[spec['layer']]=factor.fold()
            try:
                result=measure(model,selection);selection_examples+=sum(len(b[1]) for b in selection.values())
                accepted=failed is None and gate(result) and objective(result,floors)<objective(metrics,floors)
            except (ValueError,RuntimeError) as exc:
                failed=str(exc);result=None;accepted=False
            row={'stage':stage,'trial':index,'spec':spec,'metrics':result,'protected_gate':gate(result) if result else False,'failure':failed,'selected':False,'rank_after':rank_used+spec.get('rank',0),'steps_requested':budget.steps_per_trial}
            trace.append(row)
            if accepted:candidates.append((objective(result,floors),spec.get('rank',0),model,result,len(trace)-1))
        stage_steps.append(len(specs)*budget.steps_per_trial)
        if not candidates:break
        _,rank,current,metrics,trace_id=min(candidates,key=lambda c:(c[0],c[1],c[4]));trace[trace_id]['selected']=True;rank_used+=rank
    final=measure(current,selection);selection_examples+=sum(len(b[1]) for b in selection.values())
    report={'algorithm':'trained_intervention_search/v2','mode':mode,'budget':asdict(budget),'initial':initial,'final':final,'effective_floors':floors,'trace':trace,
            'rank_used_upper_bound':rank_used,'gradient_steps_total_search':gradient_steps,'gradient_examples_total_search':gradient_examples,
            'selection_examples_total_search':selection_examples,'calibration_metric_examples':calibration_metric_examples,'teacher_calls':0,'trial_count':len(trace),
            'wall_seconds':time.perf_counter()-start,'calibration_sha256':cal_hash,'selection_sha256':sel_hash,'stage_search_steps':stage_steps,
            'status':'acceptable candidate found' if objective(final,floors)[0]<=1e-9 and gate(final) else 'no acceptable candidate found',
            'worst_deficit':objective(final,floors)[0],'fraction_floors_met':sum(final[n]['accuracy']+1e-9>=floors[n] for n in names)/len(names)}
    artifact=export_artifact(output,current,ordinary.manifest['contract'],base_id=ordinary.manifest['base_id'],license=ordinary.manifest['license'],provenance={'operation':'v2 bounded intervention search','ordinary_weights_sha256':ordinary.manifest['weights_sha256'],'report':report})
    return artifact,report
