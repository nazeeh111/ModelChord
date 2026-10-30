#!/usr/bin/env python3
"""Re-run v0.2 in a new folder; a replay is not a new untouched research cohort."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys,urllib.request,zipfile,io
ROOT=Path(__file__).resolve().parents[1]

def copy_project(target):
 for name in ['src','scripts','examples','tests','docs','study','artifacts','.github']:
  shutil.copytree(ROOT/name,target/name,ignore=shutil.ignore_patterns('__pycache__','private','media'))
 for name in ['pyproject.toml','README.md','LICENSE','MANIFEST.in','requirements-macos.lock']:
  shutil.copy2(ROOT/name,target/name)
 source=ROOT/'experiments/v02';out=target/'experiments/v02';out.parent.mkdir()
 def ignore(folder,names):return [n for n in names if n in {'main','main-v1','pilots','__pycache__','raw'} or n.startswith('confirmation-') or n in {'post-confirmation-integrity.json','build-profile.json','runtime-profile.json'}]
 shutil.copytree(source,out,ignore=ignore)
 (out/'pilots').mkdir();return out

def get_data(target,out):
 # Download only known official archives, bind exact recorded bytes and members.
 manifest=json.loads((out/'data/source-manifest.json').read_text());raw=out/'data/raw';raw.mkdir()
 for dataset,download in manifest['downloads'].items():
  with urllib.request.urlopen(download['url'],timeout=30) as response:data=response.read(4*1024*1024)
  if hashlib.sha256(data).hexdigest()!=download['archive_sha256']:raise ValueError('Official '+dataset+' archive differs')
  with zipfile.ZipFile(io.BytesIO(data)) as archive:
   for member in archive.namelist():
    if dataset=='youtube' and not (member.startswith('Youtube') and member.endswith('.csv')):continue
    if dataset=='optical' and member not in {'optdigits.tra','optdigits.tes','optdigits.names','readme.txt'}:continue
    key='raw/'+member
    if key not in manifest['files']:continue
    b=archive.read(member)
    if hashlib.sha256(b).hexdigest()!=manifest['files'][key]['sha256']:raise ValueError('Archive member differs')
    (raw/member).write_bytes(b)
 sms=json.loads((target/'study/text-split.json').read_text());private=target/'study/private';private.mkdir()
 with urllib.request.urlopen('https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip',timeout=30) as response:data=response.read(2*1024*1024)
 if hashlib.sha256(data).hexdigest()!=sms['zip_sha256']:raise ValueError('Official SMS source differs')
 (private/'sms.zip').write_bytes(data)


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--replay-confirmation',action='store_true');a=p.parse_args()
 target=a.output.resolve()
 if target.exists():raise FileExistsError('Choose a fresh output; prior work is preserved')
 target.mkdir();out=copy_project(target);get_data(target,out)
 rebuilt=target/'rebuilt-text-development.npz'
 subprocess.run([sys.executable,str(target/'scripts/prepare_v2_development.py'),'--output',str(rebuilt),'--check'],cwd=target,check=True)
 print('Prepared fresh v0.2 replay workspace',target,flush=True)
 if a.prepare_only:return
 subprocess.run([sys.executable,str(out/'run_suite.py')],cwd=target,check=True)
 shutil.copy2(out/'PROTOCOL.md',out/'main/protocol-registered.md')
 subprocess.run([sys.executable,str(out/'evaluate_locked.py'),'freeze'],cwd=target,check=True)
 if a.replay_confirmation:
  print('Replaying a PUBLICLY OBSERVED cohort; this is reproduction, not new confirmation evidence.',flush=True)
  subprocess.run([sys.executable,str(out/'evaluate_locked.py'),'confirm'],cwd=target,check=True)
if __name__=='__main__':main()
