"""Typed core API with a small CLI. JSON files contain data, never executable recipes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from .artifacts import Artifact, validate_compatibility
from .merge import METHODS, merge

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modelchord",description="Build and run one static model from compatible specialists")
    parser.add_argument("--version",action="version",version="modelchord 0.1.0")
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
        elif args.command=="repair":
            from .repair import repair, Capability, BuildBudget
            caps=[Capability(**c) for c in json.loads(args.capabilities.read_text())]
            if args.data.stat().st_size>64*1024*1024:raise ValueError("Repair data exceeds 64 MiB")
            with np.load(args.data,allow_pickle=False) as data:
                cal={c.name:(torch.from_numpy(data[f"cal_x_{c.name}"].copy()).float(),torch.from_numpy(data[f"cal_y_{c.name}"].copy()).long()) for c in caps}
                select={c.name:(torch.from_numpy(data[f"x_{c.name}"].copy()).float(),torch.from_numpy(data[f"y_{c.name}"].copy()).long()) for c in caps}
            artifact,report=repair(Artifact.open(args.ordinary),cal,select,caps,args.output,budget=BuildBudget(rank=args.rank,rounds=args.rounds))
            result={"artifact":str(artifact.path),"report":report}
        else:
            from .server import serve
            serve(args.project,args.port)
            return 0
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        print(json.dumps({"error":str(exc)}),file=sys.stderr);return 2

if __name__=="__main__":
    raise SystemExit(main())
