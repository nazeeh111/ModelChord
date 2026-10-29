"""Independent text application, no base or parent adapter needed at runtime.

python examples/text_app.py artifacts/text/repaired 'Meeting at two.' 'FREE prize!'
Character representation is the versioned file beside this app.
"""
import argparse
import json
from pathlib import Path
import torch
from modelchord import Artifact
from text_features import text_features
p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('texts',nargs='+');a=p.parse_args()
model=Artifact.open(a.artifact)
if model.manifest['contract']['family']!='text':raise ValueError('Choose a text artifact')
probs=model.predict(torch.from_numpy(text_features(a.texts))).softmax(1)
classes=model.manifest['architecture']['classes']
print(json.dumps({'classes':classes,'labels':[classes[int(i)] for i in probs.argmax(1)],'probabilities':probs.tolist()},indent=2))
