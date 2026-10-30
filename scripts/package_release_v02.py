#!/usr/bin/env python3
"""Package the frozen v0.2 artifacts with explicit acceptance and attribution."""
from pathlib import Path
import hashlib,json,shutil,zipfile
root=Path(__file__).resolve().parents[1];out=root/'publication/v02/assets';out.mkdir(parents=True,exist_ok=True)
lock=json.loads((root/'experiments/v02/confirmation-lock.json').read_text());results=json.loads((root/'experiments/v02/confirmation-results.json').read_text())
for suite,path in lock['deployments'].items():
 row=next(r for r in results['rows'] if r['artifact']==path)
 with zipfile.ZipFile(out/f'modelchord-{suite}-v0.2.0.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted((root/path).iterdir()):z.write(f,f.name)
  for f in ['LICENSE','docs/ATTRIBUTION.md'] :z.write(root/f,Path(f).name)
  z.writestr('ACCEPTANCE.json',json.dumps({'version':'0.2.0','acceptance':row['acceptance'],'metrics':row['metrics'],'required':row['required'],'scope':'Frozen cohort empirical result; not a universal guarantee. Text transfer failed. Strong matched controls are not outperformed generally.'},indent=2)+'\n')
  if suite=='text':
   z.write(root/'examples/text_features.py','text_features.py')
   z.write(root/'examples/text_app.py','text_app.py')
  else:z.write(root/'examples/predict.py','predict.py')
for src,dst in [('dist/modelchord-0.2.0-py3-none-any.whl','modelchord-0.2.0-py3-none-any.whl'),('dist/modelchord-0.2.0.tar.gz','modelchord-0.2.0.tar.gz'),('experiments/v02/confirmation-results.json','confirmation-results-v0.2.0.json'),('experiments/v02/confirmation-lock.json','confirmation-lock-v0.2.0.json'),('docs/media/v02/MODELCHORD-v0.2.0.mp4','MODELCHORD-v0.2.0.mp4')]:shutil.copy2(root/src,out/dst)
(out/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n' for f in sorted(out.iterdir()) if f.is_file() and f.name!='SHA256SUMS'))
print(json.dumps({'assets':[{ 'name':f.name,'bytes':f.stat().st_size} for f in sorted(out.iterdir())]},indent=2))
