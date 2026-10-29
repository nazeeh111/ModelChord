"""Layerwise baselines, including executable TSV and activation-informed RegMean."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import math
import hashlib
import torch
from .artifacts import Artifact, canonical, export_artifact, validate_compatibility
from .models import from_spec

METHODS = {"average","task_arithmetic","ties","dare","tsv","regmean"}

def _ties(deltas: torch.Tensor, density: float) -> torch.Tensor:
    flat = deltas.flatten(1)
    count = max(1,math.ceil(flat.shape[1]*density))
    threshold = flat.abs().topk(count,dim=1).values[:,-1:]
    trimmed = torch.where(flat.abs() >= threshold,flat,0)
    sign = trimmed.sum(0).sign()
    aligned = trimmed * ((trimmed.sign()==sign) & (sign!=0))
    merged = aligned.sum(0) / (aligned.ne(0).sum(0).clamp_min(1))
    return merged.reshape(deltas.shape[1:])

def _tsv(deltas: torch.Tensor) -> torch.Tensor:
    # Task Singular Vectors, Algorithm 1; no reimplementation claim.
    n,m = deltas.shape[1:]
    rank = max(1,min(n,m)//len(deltas))
    if rank*len(deltas)>min(n,m):
        return deltas.mean(0)  # Tiny tensors cannot admit orthogonal task subspaces.
    us,ss,vs = [],[],[]
    for d in deltas:
        u,s,vh = torch.linalg.svd(d,full_matrices=False)
        us.append(u[:,:rank]); ss.append(s[:rank]); vs.append(vh[:rank].T)
    ucat,vcat = torch.cat(us,1),torch.cat(vs,1)
    p,_,qh = torch.linalg.svd(ucat,full_matrices=False); up = p @ qh
    p,_,qh = torch.linalg.svd(vcat,full_matrices=False); vp = p @ qh
    return (up * torch.cat(ss)) @ vp.T

def merge(base: Artifact, experts: list[Artifact], output: str | Path, *, method: str = "task_arithmetic",
          scale: float = 0.5, density: float = 0.2, seed: int = 0,
          calibration: list[torch.Tensor] | None = None, ridge: float = 1e-3) -> Artifact:
    compatibility = validate_compatibility(base,experts)
    if method not in METHODS:
        raise ValueError(f"Unsupported merge method: {method}")
    if not math.isfinite(scale) or not 0 <= scale <= 4 or not 0 < density <= 1 or not 0 < ridge <= 1:
        raise ValueError("Scale [0,4], density (0,1], ridge (0,1] required")
    if method == "regmean" and (calibration is None or len(calibration)!=len(experts)):
        raise ValueError("RegMean requires one permitted calibration tensor per expert")
    # Models are opened one at a time; Gram statistics stay small in this registry.
    grams: dict[str,list[torch.Tensor]] = {}
    if method == "regmean":
        for expert,x in zip(experts,calibration):
            if x.ndim != 2 or not torch.isfinite(x).all() or len(x)==0:
                raise ValueError("Invalid calibration inputs")
            with torch.no_grad():
                acts = expert.model().activations(x)
            for k,a in acts.items():
                # Bias is handled exactly by augmenting the activation with a constant.
                a = torch.cat([a,torch.ones(len(a),1)],1).double()
                g = a.T@a / len(a)
                # Original RegMean reduces off-diagonal covariance to stabilize merging.
                g = 0.5*g + 0.5*torch.diag(torch.diag(g))
                grams.setdefault(k,[]).append(g)
    state = {}
    generator = torch.Generator().manual_seed(seed)
    for key in base.manifest["tensors"]:
        if method == "regmean":
            if not key.endswith("weight"):
                continue
            bias = key.replace("weight","bias")
            values = [torch.cat([e.tensor(key),e.tensor(bias)[:,None]],1).double() for e in experts]
            gs = grams[key]
            total = sum(gs)
            penalty = ridge*max(float(total.diag().mean()),1e-8)
            regularized = total + penalty*torch.eye(total.shape[0],dtype=torch.float64)
            # Ridge toward base (rather than zero), with documented strength.
            b = torch.cat([base.tensor(key),base.tensor(bias)[:,None]],1).double()
            rhs = sum(w@g for w,g in zip(values,gs)) + penalty*b
            result = torch.linalg.solve(regularized,rhs.T).T.float()
            state[key],state[bias] = result[:,:-1],result[:,-1]
            continue
        b = base.tensor(key)
        deltas = torch.stack([e.tensor(key)-b for e in experts])
        if method == "average":
            update = deltas.mean(0)
        elif method == "task_arithmetic":
            update = deltas.sum(0)
        elif method == "ties":
            update = _ties(deltas,density)
        elif method == "dare":
            mask = torch.rand(deltas.shape,generator=generator)<density
            update = (deltas*mask/density).sum(0)
        else:
            update = _tsv(deltas) if b.ndim==2 else deltas.mean(0)
        state[key] = b + scale*update
    model = from_spec(base.manifest["architecture"])
    model.load_state_dict(state)
    return export_artifact(output,model,base.manifest["contract"],base_id=base.manifest["base_id"],license=base.manifest["license"],
        provenance={"operation":"merge","method":method,"scale":scale,"density":density,"seed":seed,"ridge":ridge,
                    "base_weights":base.manifest["weights_sha256"],"expert_weights":[e.manifest["weights_sha256"] for e in experts],
                    "source_manifests":[{"sha256":hashlib.sha256(canonical(a.manifest)).hexdigest(),
                                         "license":a.manifest["license"],"provenance":a.manifest["provenance"]} for a in [base,*experts]],
                    "compatibility":compatibility,"regmean_offdiagonal_factor":0.5 if method=="regmean" else None,
                    "calibration_inputs_sha256":[hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for x in calibration] if method=="regmean" else [],
                    "calibration_examples":sum(len(x) for x in calibration) if method=="regmean" else 0})
