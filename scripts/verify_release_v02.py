#!/usr/bin/env python3
"""Verify anonymous v0.2 assets and inference with both source trees unavailable."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,tempfile,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
BASE='https://github.com/nazeeh111/ModelChord/releases/download/v0.2.0/'
NAMES=['SHA256SUMS','MODELCHORD-v0.2.0.mp4','confirmation-lock-v0.2.0.json','confirmation-results-v0.2.0.json','modelchord-0.2.0-py3-none-any.whl','modelchord-0.2.0.tar.gz','modelchord-text-v0.2.0.zip','modelchord-vision-v0.2.0.zip']

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--downloads',type=Path,required=True);p.add_argument('--python',type=Path,required=True);p.add_argument('--clone',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.downloads.exists():raise FileExistsError('Choose a fresh anonymous-download folder')
 a.downloads.mkdir(parents=True)
 for name in NAMES:
  with urllib.request.urlopen(BASE+name,timeout=60) as response:data=response.read()
  (a.downloads/name).write_bytes(data)
 expected={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (a.downloads/'SHA256SUMS').read_text().splitlines()}
 assert set(expected)==set(NAMES)-{'SHA256SUMS'}
 hashes={name:digest(a.downloads/name) for name in NAMES}
 assert all(hashes[name]==sha for name,sha in expected.items())
 local=ROOT/'publication/v02/assets'
 assert all((local/name).read_bytes()==(a.downloads/name).read_bytes() for name in NAMES)
 # Reinstall the actual anonymously downloaded wheel into a dependency-resolved
 # temporary environment, outside both source repositories.
 subprocess.run(['uv','pip','install','--python',str(a.python),'--no-deps','--reinstall',str(a.downloads/'modelchord-0.2.0-py3-none-any.whl')],check=True)
 version=subprocess.check_output([str(a.python),'-I','-c','import modelchord;print(modelchord.__version__)'],text=True).strip();assert version=='0.2.0'
 with tempfile.TemporaryDirectory(prefix='modelchord-v02-release-proof-') as tmp:
  proof=Path(tmp)
  for suite in ['vision','text']:
   destination=proof/suite;destination.mkdir()
   with zipfile.ZipFile(a.downloads/f'modelchord-{suite}-v0.2.0.zip') as z:
    names=['manifest.json','model.safetensors']+(['text_features.py'] if suite=='text' else [])
    for name in names:(destination/name).write_bytes(z.read(name))
  denied=[ROOT.resolve(),a.clone.resolve(),Path.home()/'.cache',Path.home()/'.ollama',Path.home()/'Library/Caches']
  profile=proof/'sandbox.sb';profile.write_text('(version 1)\n(allow default)\n(deny network*)\n'+'\n'.join('(deny file-read* (subpath '+json.dumps(str(path))+'))' for path in denied)+'\n')
  worker=proof/'worker.py';worker.write_text('''import sys,json,socket,importlib.util
from pathlib import Path
import torch
from modelchord import Artifact
torch.set_num_threads(1)
root=Path(sys.argv[1]);denied=json.loads(sys.argv[2]);proofs={}
for path in denied:
 try:list(Path(path).iterdir())
 except PermissionError:continue
 raise AssertionError('Denied source/cache path was readable: '+path)
try:
 s=socket.socket();s.settimeout(1);s.connect(('1.1.1.1',443))
except OSError:pass
else:raise AssertionError('Network available')
for suite in ['vision','text']:
 artifact=Artifact.open(root/suite)
 if suite=='text':
  spec=importlib.util.spec_from_file_location('features',root/suite/'text_features.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  x=torch.from_numpy(m.text_features(['Meeting at two.','WIN a FREE prize!']))
 else:
  x=torch.zeros(2,64);x[1,::3]=1
 y=artifact.predict(x);assert not torch.equal(y[0],y[1])
 proofs[suite]={'changed_input':True,'logits':y.tolist(),'classes':artifact.manifest['architecture']['classes'],'weights_sha256':artifact.manifest['weights_sha256']}
print(json.dumps({'source_and_clone_blocked':True,'caches_blocked':True,'network_blocked':True,'artifacts':proofs}))
''')
  env={'PATH':'/usr/bin:/bin','HOME':str(proof),'TMPDIR':str(proof),'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
  run=subprocess.run(['/usr/bin/sandbox-exec','-f',str(profile),str(a.python),'-I',str(worker),str(proof),json.dumps([str(p) for p in denied])],cwd=proof,env=env,capture_output=True,text=True,timeout=180)
  evidence={'version':version,'anonymous_asset_sha256':hashes,'all_download_bytes_match_local_assets':True,'sandbox_denied_paths':[str(p) for p in denied],'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
  if run.returncode==0:evidence['observed']=json.loads(run.stdout)
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence,indent=2));assert run.returncode==0
if __name__=='__main__':main()
