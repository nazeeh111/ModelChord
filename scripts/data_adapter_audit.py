"""Verify safe adapter factors reconstruct each exported text specialist."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json,torch
from safetensors import safe_open
from modelchord import Artifact
from modelchord.artifacts import canonical,digest
root=Path(__file__).resolve().parents[1];base=Artifact.open(root/'artifacts/text/base');rows=[]
for task in ['uppercase','leet','underscores']:
 meta=json.loads((root/f'artifacts/text/adapter-{task}.json').read_text());p=root/f'artifacts/text/adapter-{task}.safetensors'
 assert digest(p)==meta['adapter_sha256']
 model=base.model()
 with safe_open(p,framework='pt',device='cpu') as f,torch.no_grad():
  for i,layer in enumerate(model.layers):layer.weight.add_(f.get_tensor(f'layers.{i}.b')@f.get_tensor(f'layers.{i}.a'))
 folded=Artifact.open(root/f'artifacts/text/expert-{task}').model()
 sample=torch.stack([torch.zeros(256),torch.ones(256)/16,torch.linspace(-1,1,256)])
 with torch.no_grad():error=float((model(sample)-folded(sample)).abs().max())
 assert error<1e-6
 rows.append({'task':task,'adapter_sha256':digest(p),'folded_weights_sha256':meta['folded_expert_weights_sha256'],'changed_inputs':3,'max_abs_logit_error':error,'base_path_only_during_audit':True,'deployed_form':'dense static folded weights; parent base and adapter files absent at inference'})
(root/'study/text-adapter-audit.json').write_bytes(canonical(rows)+b'\n');print(json.dumps(rows,indent=2))
