"""Pre-confirmation specialist-training diagnostic; all attempted checkpoints retained."""
from copy import deepcopy
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_study as s
import torch
from torch.nn import functional as F
from modelchord import Artifact

x,y=s.vision_data();spl=s.load(s.ROOT/'study/vision-split.json')['indices'];data=s.partitions(x,y,spl)
rows=[]
for seed in [11,29]:
 base=Artifact.open(s.ROOT/f'artifacts/vision-seed{seed}/base');m=base.model();opt=torch.optim.Adam(m.parameters(),lr=.001)
 for epoch in range(400):
  opt.zero_grad();loss=F.cross_entropy(m(data['train']['inverted'][0]),data['train']['inverted'][1]);loss.backward();opt.step()
  if (epoch+1)%100==0:
   a=s.export(s.ROOT/f'artifacts/vision-seed{seed}/expert-inverted-diagnostic-{epoch+1}',m,base.manifest['contract'],base.manifest['base_id'],**s.provenance('vision',seed,s.digest(s.ROOT/'study/vision-split.json'),task='inverted',operation='pre-confirmation weak-specialist diagnostic',training={'epochs':epoch+1,'optimizer':'Adam','lr':.001}))
   metrics=s.classification(m,*data['selection']['inverted']);r={'seed':seed,'epochs':epoch+1,'artifact':str(a.path.relative_to(s.ROOT)),'selection':metrics};rows.append(r);s.write(s.ROOT/'study/inversion-training-diagnostic.json',rows);print(r,flush=True)
