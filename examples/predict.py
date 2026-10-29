"""Independent application: only an exported artifact and MODELCHORD are needed.

python examples/predict.py artifacts/vision-seed11-v2/repair-strongest --input image.json
JSON input: [batch,64] floats, 8x8 row-major grayscale, values normalized /16.
Use the actual path in study/deployment-selection.json. No expert or base is loaded here.
"""
import argparse
import json
from pathlib import Path
import torch
from modelchord import Artifact
p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('--input',type=Path)
a=p.parse_args();model=Artifact.open(a.artifact)
x=torch.tensor(json.loads(a.input.read_text()),dtype=torch.float32) if a.input else torch.stack([torch.zeros(64),torch.ones(64)])
probs=model.predict(x).softmax(1)
classes=model.manifest['architecture']['classes']
print(json.dumps({'classes':classes,'labels':[classes[int(i)] for i in probs.argmax(1)],'probabilities':probs.tolist()},indent=2))
