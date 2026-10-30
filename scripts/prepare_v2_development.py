#!/usr/bin/env python3
"""Reconstruct v2 text development arrays from frozen SMS source-ID roles only."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'examples')]
import numpy as np
import run_study as s

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
 if a.output.exists():raise FileExistsError('Use a fresh output')
 texts,labels,zip_hash,_=s.sms_data();frozen=s.load(ROOT/'study/text-split.json')
 if zip_hash!=frozen['zip_sha256']:raise ValueError('SMS source hash differs')
 by_hash={hashlib.sha256(t.encode()).hexdigest():i for i,t in enumerate(texts)}
 roles={part:[by_hash[h] for h in frozen['ids'][part]] for part in ['calibration','selection']}
 assert not set(roles['calibration'])&set(roles['selection'])
 old_confirm=set(frozen['ids']['confirmation'])
 assert not old_confirm&(set(frozen['ids']['calibration'])|set(frozen['ids']['selection']))
 batches=s.text_batches(texts,labels,roles,['calibration','selection']);arrays={}
 for prefix,part in [('cal_','calibration'),('','selection')]:
  for task,(x,y) in batches[part].items():arrays[f'{prefix}x_{task}']=x.numpy();arrays[f'{prefix}y_{task}']=y.numpy()
 if a.check:
  with np.load(ROOT/'experiments/v02/text-development.npz',allow_pickle=False) as original:
   assert set(original.files)==set(arrays)
   assert all(np.array_equal(original[name],array) for name,array in arrays.items())
 np.savez(a.output,**arrays)
 print(json.dumps({'array_count':len(arrays),'source_id_roles':'frozen calibration/selection only','old_confirmation_excluded':True,'exact_frozen_array_match':a.check,'output_sha256':s.digest(a.output)},indent=2))
if __name__=='__main__':main()
