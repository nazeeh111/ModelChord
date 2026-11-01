"""Typed core API with a small CLI. JSON files contain data, never executable recipes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import math
import numpy as np
import torch
from .artifacts import Artifact, validate_compatibility
from .merge import METHODS, merge

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modelchord",description="Build and run one static model from compatible specialists")
    from . import __version__
    parser.add_argument("--version",action="version",version=f"modelchord {__version__}")
    sub=parser.add_subparsers(dest="command",required=True)
    inspect=sub.add_parser("inspect");inspect.add_argument("artifact",type=Path)
    check=sub.add_parser("check"); check.add_argument("base",type=Path);check.add_argument("experts",nargs="+",type=Path)
    build=sub.add_parser("merge");build.add_argument("base",type=Path);build.add_argument("experts",nargs="+",type=Path)
    build.add_argument("--output",required=True,type=Path);build.add_argument("--method",choices=sorted(METHODS-{"regmean"}),default="task_arithmetic")
    build.add_argument("--scale",type=float,default=.5);build.add_argument("--density",type=float,default=.2)
    fix=sub.add_parser("repair",help="Attempt bounded low-rank repair with permitted disjoint data")
    fix.add_argument("ordinary",type=Path);fix.add_argument("--data",type=Path,required=True,help="Safe npz: cal_x_NAME/cal_y_NAME and x_NAME/y_NAME")
    fix.add_argument("--capabilities",type=Path,required=True,help="JSON array of capability objects")
    fix.add_argument("--output",type=Path,required=True);fix.add_argument("--rank",type=int,default=8);fix.add_argument("--rounds",type=int,default=4)
    trained=sub.add_parser("repair-v2",help="Train bounded factor interventions with disjoint selection gates")
    trained.add_argument("ordinary",type=Path);trained.add_argument("--data",type=Path,required=True)
    trained.add_argument("--capabilities",type=Path,required=True);trained.add_argument("--output",type=Path,required=True)
    trained.add_argument("--rank",type=int,default=8);trained.add_argument("--stages",type=int,default=4)
    trained.add_argument("--steps",type=int,default=30);trained.add_argument("--planning-margin",type=float,default=.03)
    predict=sub.add_parser("predict");predict.add_argument("artifact",type=Path);predict.add_argument("inputs",type=Path,help="JSON array [batch,input_dim]")
    serve=sub.add_parser("serve");serve.add_argument("--project",type=Path,default=Path.cwd());serve.add_argument("--port",type=int,default=8765)
    args=parser.parse_args(argv)
    torch.set_num_threads(2)
    try:
        if args.command=="inspect": result=Artifact.open(args.artifact).manifest
        elif args.command=="check":result=validate_compatibility(Artifact.open(args.base),[Artifact.open(p) for p in args.experts])
        elif args.command=="merge":
            artifact=merge(Artifact.open(args.base),[Artifact.open(p) for p in args.experts],args.output,method=args.method,scale=args.scale,density=args.density)
            result={"artifact":str(artifact.path),"manifest":artifact.manifest}
        elif args.command=="predict":
            if args.inputs.stat().st_size>4*1024*1024:raise ValueError("Input JSON exceeds 4 MiB")
            x=torch.tensor(json.loads(args.inputs.read_text()),dtype=torch.float32)
            artifact=Artifact.open(args.artifact);probs=artifact.predict(x).softmax(1)
            classes=artifact.manifest["architecture"]["classes"]
            result={"labels":[classes[int(i)] for i in probs.argmax(1)],"classes":classes,"probabilities":probs.tolist()}
        elif args.command in {"repair","repair-v2"}:
            from .repair import repair, Capability, BuildBudget
            caps=[Capability(**c) for c in json.loads(args.capabilities.read_text())]
            if args.data.stat().st_size>64*1024*1024:raise ValueError("Repair data exceeds 64 MiB")
            with np.load(args.data,allow_pickle=False) as data:
                cal={c.name:(torch.from_numpy(data[f"cal_x_{c.name}"].copy()),torch.from_numpy(data[f"cal_y_{c.name}"].copy())) for c in caps}
                select={c.name:(torch.from_numpy(data[f"x_{c.name}"].copy()),torch.from_numpy(data[f"y_{c.name}"].copy())) for c in caps}
            if args.command=="repair":
                cal={n:(x.float(),y.long()) for n,(x,y) in cal.items()}
                select={n:(x.float(),y.long()) for n,(x,y) in select.items()}
            if args.command=="repair-v2":
                from .repair_v2 import search, SearchBudget
                if not 0<=args.planning_margin<=.1:raise ValueError("Planning margin must be in [0,.1]")
                if any(not isinstance(c.floor,(int,float)) or not math.isfinite(c.floor) or not 0<=c.floor<=1 for c in caps):raise ValueError("Finite requested floors in [0,1] required")
                raw_floors={c.name:c.floor for c in caps}
                if not 0<=args.rank<=64 or not 0<=args.stages<=8:raise ValueError("Rank 0..64 and stages 0..8 required")
                if not 1<=args.steps<=64:raise ValueError("Steps per trial must be 1..64")
                stages=min(args.stages,args.rank//2)
                margin=args.planning_margin if stages else 0.
                caps=[Capability(c.name,min(1.,c.floor+margin),c.protected,c.max_regression) for c in caps]
                artifact,report=search(Artifact.open(args.ordinary),cal,select,caps,args.output,budget=SearchBudget(rank=args.rank,stages=stages,steps_per_trial=args.steps))
                report['requested_floors']=raw_floors;report['planning_margin']=margin
                report['effective_selection_floors']=report['effective_floors']
                report['required_fraction_floors_met']=sum(report['final'][n]['accuracy']>=v for n,v in raw_floors.items())/len(raw_floors)
                (artifact.path/'build-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            else:
                artifact,report=repair(Artifact.open(args.ordinary),cal,select,caps,args.output,budget=BuildBudget(rank=args.rank,rounds=args.rounds))
            result={"artifact":str(artifact.path),"report":report}
        else:
            from .server import serve
            serve(args.project,args.port)
            return 0
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
        print(json.dumps({"error":str(exc)}),file=sys.stderr);return 2

if __name__=="__main__":
    raise SystemExit(main())
