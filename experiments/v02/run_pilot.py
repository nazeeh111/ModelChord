from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import numpy as np,torch
from modelchord import Artifact,Capability
from modelchord.artifacts import canonical
from repair_search import search,reference_search,SearchBudget

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=11);p.add_argument('--steps',type=int,default=30);p.add_argument('--lr',type=float,default=.02);p.add_argument('--mode',default='deficit_factor');p.add_argument('--stages',type=int,default=3);p.add_argument('--suite',choices=['vision','text'],default='vision');p.add_argument('--margin',type=float,default=0.0);a=p.parse_args()
 torch.set_num_threads(2);folder=ROOT/'experiments/v02/pilots'/f'{a.suite}-seed{a.seed}-{a.mode}-steps{a.steps}-lr{a.lr}-margin{a.margin}'
 if folder.exists():raise FileExistsError('Keep existing attempts immutable')
 folder.mkdir();start={'seed':a.seed,'mode':a.mode,'steps':a.steps,'lr':a.lr,'margin':a.margin,'repair_code_sha256':hashlib.sha256((folder.parents[1]/'repair_search.py').read_bytes()).hexdigest(),'started':time.time(),'confirmation_accessed':False};(folder/'registered.json').write_bytes(canonical(start));(folder/'repair-source.py').write_bytes((folder.parents[1]/'repair_search.py').read_bytes())
 with np.load(ROOT/('study/workbench.npz' if a.suite=='vision' else 'experiments/v02/text-development.npz'),allow_pickle=False) as d:
  names=['clean','left','right','inverted'] if a.suite=='vision' else ['clean','uppercase','leet','underscores'];cal={n:(torch.tensor(d[f'cal_x_{n}']),torch.tensor(d[f'cal_y_{n}'])) for n in names};sel={n:(torch.tensor(d[f'x_{n}']),torch.tensor(d[f'y_{n}'])) for n in names}
 ordinary=Artifact.open(ROOT/f'artifacts/vision-seed{a.seed}-v2/regmean-0') if a.suite=='vision' else Artifact.open(ROOT/'artifacts/text/ordinary-3');caps=[Capability(n,((.88 if n=='clean' else .80) if a.suite=='vision' else (.75 if n=='clean' else .72))+a.margin,n=='clean',.02 if a.suite=='vision' else .03) for n in names]
 function=reference_search if a.mode in ['joint','fixed_factor'] else search
 artifact,report=function(ordinary,cal,sel,caps,folder/'model',mode=a.mode,budget=SearchBudget(steps_per_trial=a.steps,factor_lr=a.lr),**({'stages':a.stages} if a.mode in ['joint','fixed_factor'] else {}))
 (folder/'report.json').write_bytes(canonical(report));print(json.dumps({k:report[k] for k in ['status','final','worst_deficit','rank_used_upper_bound','gradient_examples_total_search','selection_examples_total_search','wall_seconds']},indent=2),flush=True)
if __name__=='__main__':main()
