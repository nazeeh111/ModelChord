#!/usr/bin/env python3
"""Run installed MODELCHORD with parent models, source, caches and network denied.

Launch from an independent venv after copying only the chosen final artifact here.
This runner prints observed outputs and refuses unchanged inference.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

p=argparse.ArgumentParser();p.add_argument('--python',required=True,type=Path);p.add_argument('--artifact',required=True,type=Path)
p.add_argument('--deny',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--timeout',type=int,default=180)
a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='modelchord-independent-') as tmp:
    root=Path(tmp);artifact=root/'model';artifact.mkdir()
    for name in ['manifest.json','model.safetensors']:
        (artifact/name).write_bytes((a.artifact/name).read_bytes())
    worker=root/'worker.py'
    worker.write_text('''import json, socket, sys\nfrom pathlib import Path\nprint("worker: importing torch",file=sys.stderr,flush=True)\nimport torch\nprint("worker: importing modelchord",file=sys.stderr,flush=True)\nfrom modelchord import Artifact\ntorch.set_num_threads(1)\na=Artifact.open(Path(sys.argv[1]))\nx=torch.zeros(2,a.manifest["architecture"]["input_dim"])\nx[1,::3]=1\ny=a.predict(x)\nassert not torch.equal(y[0],y[1]), "Inference did not respond to changed inputs"\nnetwork_blocked=False\ntry:\n s=socket.socket();s.settimeout(1);s.connect(("1.1.1.1",443))\nexcept OSError:\n network_blocked=True\nassert network_blocked, "Network was available"\nparents_blocked=False\ntry: list(Path(sys.argv[2]).iterdir())\nexcept PermissionError: parents_blocked=True\nassert parents_blocked,"Parent project was readable"\nprint(json.dumps({"changed_input":True,"logits":y.tolist(),"classes":a.manifest["architecture"]["classes"],"network_blocked":network_blocked,"parent_project_blocked":parents_blocked,"artifact_bytes":sum(p.stat().st_size for p in Path(sys.argv[1]).iterdir()),"torch":torch.__version__}))\n''')
    if not Path('/usr/bin/sandbox-exec').exists():
        raise SystemExit('This strict verification runner currently requires macOS sandbox-exec; use an isolated networkless container on Linux')
    profile=root/'sandbox.sb'
    # The project includes every parent expert. Deny all project reads, not merely
    # a network monkeypatch; the installed wheel and copied artifact live elsewhere.
    deny=json.dumps(str(a.deny.resolve()))
    home=Path.home()
    cache_paths=[home/'.cache',home/'.cache/huggingface',home/'.ollama',home/'Library/Caches']
    clauses='\n'.join('(deny file-read* (subpath '+json.dumps(str(c))+'))' for c in cache_paths)
    profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* (subpath '+deny+'))\n'+clauses+'\n')
    command=['/usr/bin/sandbox-exec','-f',str(profile),str(a.python.absolute()),'-I',str(worker),str(artifact),str(a.deny.resolve())]
    env={'PATH':'/usr/bin:/bin','HOME':str(root),'TMPDIR':str(root),'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    try:
        result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=a.timeout)
    except subprocess.TimeoutExpired as exc:
        evidence={'returncode':124,'failure':'timeout','timeout_seconds':a.timeout,'stdout':str(exc.stdout),'stderr':str(exc.stderr)}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(evidence,indent=2)+'\n')
        print(json.dumps(evidence,indent=2));raise SystemExit(124)
    evidence={'command_policy':'Independent wheel install; no user site/PYTHONPATH; copied artifact only; OS denies source/parent project, caches and all network',
              'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    if result.returncode==0:evidence['observed']=json.loads(result.stdout)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))
    raise SystemExit(result.returncode)
