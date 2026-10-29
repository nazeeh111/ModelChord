#!/usr/bin/env python3
"""Reproduce in a fresh output folder, preserving the released study unchanged."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--confirm',action='store_true',help='Also run one confirmation after training and freeze');p.add_argument('--prepare-only',action='store_true',help='Prepare verified public source/data without training');a=p.parse_args()
source=Path(__file__).resolve().parents[1];target=a.output.resolve()
if target.exists():raise SystemExit('Choose a fresh output folder; existing work is never overwritten')
target.mkdir(parents=True)
for name in ['src','scripts','examples','docs']:
    shutil.copytree(source/name,target/name,ignore=shutil.ignore_patterns('__pycache__','media'))
study=target/'study';(study/'private').mkdir(parents=True)
(study/'recipe-registry.json').write_text('[]\n')
# Public official source, downloaded afresh. No user-private records are copied.
url='https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
expected='1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3'
with urllib.request.urlopen(url,timeout=30) as response:data=response.read(2*1024*1024)
if hashlib.sha256(data).hexdigest()!=expected:raise SystemExit('Official SMS source differs from registered SHA256; verify dataset version before proceeding')
(study/'private/sms.zip').write_bytes(data)
print('Prepared verified public source/data in',target,flush=True)
if a.prepare_only:raise SystemExit(0)
command=[sys.executable,str(target/'scripts/run_study.py'),'--phase','all' if a.confirm else 'train']
raise SystemExit(subprocess.run(command,cwd=target).returncode)
