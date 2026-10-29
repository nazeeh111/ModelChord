"""Evaluator extension: class-balanced accuracy on present labeled classes.

Pass balanced_classification as repair(..., evaluator=balanced_classification).
Floors then mean mean per-class recall on the declared batch, not raw accuracy.
"""
import torch
from modelchord.evaluate import classification

@torch.no_grad()
def balanced_classification(model, x, y):
    result=classification(model,x,y)
    prediction=model(x).argmax(1)
    recalls=[prediction[y==label].eq(label).float().mean() for label in y.unique()]
    result['accuracy']=float(torch.stack(recalls).mean())
    result['score_name']='balanced accuracy over present classes'
    return result
