"""Deficit-directed projected corrections with a validation gate and rank budget.

Data-assisted optimization, not training-free merging. Weight gradients propose
subspaces; controlled interventions and held-out selection determine acceptance.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import hashlib
from typing import Any, Callable
import torch
from torch.nn import functional as F
from .artifacts import Artifact, export_artifact
from .evaluate import classification
from .models import from_spec

@dataclass(frozen=True)
class Capability:
    name: str
    floor: float
    protected: bool = False
    max_regression: float = 0.01

@dataclass(frozen=True)
class BuildBudget:
    rank: int = 8
    rounds: int = 4
    candidate_limit: int = 72
    learning_rates: tuple[float,...] = (0.01,0.04,0.12)
    ranks: tuple[int,...] = (1,2)

Batch = tuple[torch.Tensor,torch.Tensor]

def repair(ordinary: Artifact, calibration: dict[str,Batch], selection: dict[str,Batch],
           capabilities: list[Capability], output: str | Path, *, budget: BuildBudget = BuildBudget(),
           prioritize: bool = True, allocate: bool = True, protect: bool = True,
           correction: bool = True, evaluator: Callable = classification) -> tuple[Artifact,dict[str,Any]]:
    names = [c.name for c in capabilities]
    if len(set(names))!=len(names) or not names or set(names)!=set(calibration) or set(names)!=set(selection):
        raise ValueError("Capabilities and both disjoint evaluation mappings must agree")
    if any(not 0<=c.floor<=1 or not 0<=c.max_regression<=1 for c in capabilities):
        raise ValueError("Floors and permitted regressions must be in [0,1]")
    if not 0<=budget.rank<=64 or not 0<=budget.rounds<=32 or not 0<=budget.candidate_limit<=2048:
        raise ValueError("Build budget exceeds bounded runtime policy")
    if any(r<1 or r>64 for r in budget.ranks) or any(not math.isfinite(lr) or not 0<lr<=2 for lr in budget.learning_rates):
        raise ValueError("Invalid candidate ranks or learning rates")
    def batch_hash(batch):
        h = hashlib.sha256()
        for tensor in batch:
            h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    data_hashes = {"calibration":{n:batch_hash(calibration[n]) for n in names},
                   "selection":{n:batch_hash(selection[n]) for n in names}}
    if set(data_hashes["calibration"].values()) & set(data_hashes["selection"].values()):
        raise ValueError("Calibration and selection batches must be disjoint; identical batches supplied")
    model = ordinary.model()
    initial = {n:evaluator(model,*selection[n]) for n in names}
    effective = {c.name:max(c.floor,initial[c.name]["accuracy"]-c.max_regression) if c.protected and protect else c.floor for c in capabilities}
    def measure(m): return {n:evaluator(m,*selection[n]) for n in names}
    def objective(metrics):
        deficits = [max(0.,effective[n]-metrics[n]["accuracy"]) for n in names]
        return (max(deficits),sum(deficits),sum(metrics[n]["cross_entropy"] for n in names)/len(names))
    def protected_ok(metrics):
        return all(not(c.protected and protect) or metrics[c.name]["accuracy"]+1e-9 >= initial[c.name]["accuracy"]-c.max_regression for c in capabilities)
    current = initial
    trace = []; spent_rank = 0; attempts = 0; gradient_examples = 0; validation_examples = sum(len(selection[n][1]) for n in names)
    calibration_evaluation_examples = 0
    for round_id in range(budget.rounds if correction else 0):
        if objective(current)[0]<=1e-9 or spent_rank>=budget.rank or attempts>=budget.candidate_limit:
            break
        # Priority follows measured functional deficits. Satisfactory protected tasks
        # remain in the acceptance gate, rather than dominating an average loss.
        cal_metrics = {n:evaluator(model,*calibration[n]) for n in names}
        calibration_evaluation_examples += sum(len(calibration[n][1]) for n in names)
        target = max(names,key=lambda n:effective[n]-cal_metrics[n]["accuracy"])
        train_names = [target] if prioritize else names
        model.zero_grad(set_to_none=True)
        loss = sum(F.cross_entropy(model(calibration[n][0]),calibration[n][1]) for n in train_names)/len(train_names)
        loss.backward()
        gradient_examples += sum(len(calibration[n][1]) for n in train_names)
        state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        candidates=[]
        for name,param in model.named_parameters():
            if param.ndim!=2 or param.grad is None: continue
            gradient = param.grad.detach()
            u,s,vh = torch.linalg.svd(gradient,full_matrices=False)
            for rank in (budget.ranks if allocate else (1,)):
                if rank+spent_rank>budget.rank or rank>min(gradient.shape):continue
                direction = (u[:,:rank]*s[:rank])@vh[:rank]
                norm = float(direction.norm())
                if norm<1e-12:continue
                # Absolute Frobenius scaling, never expert-minus-base normalization.
                direction = direction/norm
                for lr in budget.learning_rates:
                    if attempts>=budget.candidate_limit:break
                    trial = from_spec(model.spec)
                    trial.load_state_dict(state)
                    with torch.no_grad():
                        dict(trial.named_parameters())[name].add_(direction,alpha=-lr)
                    trial_metrics=measure(trial)
                    validation_examples+=sum(len(selection[n][1]) for n in names)
                    attempts+=1
                    gate=protected_ok(trial_metrics)
                    entry={"round":round_id,"target":target,"layer":name,"rank":rank,"step":lr,"objective":list(objective(trial_metrics)),
                           "protected_gate":gate,"metrics":trial_metrics,"accepted":False}
                    trace.append(entry)
                    if gate and objective(trial_metrics)<objective(current):
                        candidates.append((objective(trial_metrics),rank,trial,trial_metrics,len(trace)-1))
        if not candidates:break
        _,rank,model,current,index=min(candidates,key=lambda c:(c[0],c[1],c[4]))
        trace[index]["accepted"]=True
        spent_rank+=rank
    metrics = measure(model)
    validation_examples += sum(len(selection[n][1]) for n in names)
    all_met=all(metrics[n]["accuracy"]+1e-9>=effective[n] for n in names) and protected_ok(metrics)
    report={"algorithm":"deficit_projected_repair/v1","data_assisted":True,"status":"acceptable candidate found" if all_met else "no acceptable candidate found",
            "capabilities":[asdict(c) for c in capabilities],"effective_selection_floors":effective,"initial":initial,"final":metrics,
            "worst_deficit":objective(metrics)[0],"fraction_floors_met":sum(metrics[n]["accuracy"]+1e-9>=effective[n] for n in names)/len(names),
            "rank_used_upper_bound":spent_rank,"budget":asdict(budget),"attempts":attempts,"gradient_examples":gradient_examples,
            "validation_examples":validation_examples,"teacher_calls":0,"trace":trace,
            "calibration_evaluation_examples":calibration_evaluation_examples,
            "examples_evaluated_total":validation_examples+calibration_evaluation_examples+gradient_examples,
            "data_sha256":data_hashes,"split_requirement":"Caller must also verify original source IDs are disjoint, including transformed views",
            "ablations":{"prioritize":prioritize,"allocate":allocate,"protect":protect,"correction":correction}}
    artifact=export_artifact(output,model,ordinary.manifest["contract"],base_id=ordinary.manifest["base_id"],license=ordinary.manifest["license"],
        provenance={"operation":"repair","ordinary_weights":ordinary.manifest["weights_sha256"],"report":report})
    return artifact,report
