"""Evaluator extension contract: a named batch maps to measured classification metrics."""
from __future__ import annotations
from typing import Protocol
import torch
from .models import MLP

class Evaluator(Protocol):
    def __call__(self, model: MLP, x: torch.Tensor, y: torch.Tensor) -> dict[str,float]: ...

@torch.no_grad()
def classification(model: MLP, x: torch.Tensor, y: torch.Tensor) -> dict[str,float]:
    if x.ndim!=2 or y.ndim!=1 or len(x)!=len(y) or len(y)==0 or not torch.isfinite(x).all():
        raise ValueError("Expected finite nonempty x and aligned 1D labels")
    logits = model(x)
    if not torch.isfinite(logits).all():
        raise ValueError("Evaluation produced nonfinite logits")
    if y.dtype!=torch.long or y.min()<0 or y.max()>=logits.shape[1]:
        raise ValueError("Invalid labels")
    probs = logits.softmax(1)
    confidence,pred = probs.max(1)
    correct = pred.eq(y)
    ece = 0.
    bin_ids = (confidence * 10).long().clamp(max=9)
    for bin_id in range(10):
        mask = bin_ids == bin_id
        if mask.any():
            ece += float(mask.float().mean()*(correct[mask].float().mean()-confidence[mask].mean()).abs())
    return {"accuracy":float(correct.float().mean()),"cross_entropy":float(torch.nn.functional.cross_entropy(logits,y)),
            "ece_10":ece,"examples":len(y)}
