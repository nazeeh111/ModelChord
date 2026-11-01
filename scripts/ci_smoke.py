#!/usr/bin/env python3
"""Run a public artifact from a separate cwd with only its own weights present."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import torch
from modelchord import Artifact
root=Path(__file__).resolve().parents[1]
v2=root/'experiments/v02/confirmation-lock.json'
config=({k:{'artifact':v} for k,v in json.loads(v2.read_text())['deployments'].items()} if v2.exists() else json.loads((root/'study/deployment-selection.json').read_text()))
for suite in ['vision','text']:
    original=Artifact.open(root/config[suite]['artifact'])
    with tempfile.TemporaryDirectory(prefix='modelchord-ci-') as tmp:
        tmp=Path(tmp);shutil.copytree(original.path,tmp/'model')
        # -I ignores user site and PYTHONPATH. The package must already be installed.
        code='''import json,sys,torch\nfrom modelchord import Artifact\na=Artifact.open(sys.argv[1]);d=a.manifest['architecture']['input_dim'];x=torch.zeros(2,d);x[1,::3]=1;z=a.predict(x);assert not torch.equal(z[0],z[1]);print(json.dumps({'changed_input':True,'shape':list(z.shape)}))'''
        result=subprocess.run([sys.executable,'-I','-c',code,str(tmp/'model')],cwd=tmp,check=True,capture_output=True,text=True)
        print(suite,result.stdout.strip())
