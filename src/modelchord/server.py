"""Loopback-only workbench. Local artifacts are addressed by a fixed catalog."""
from __future__ import annotations
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse, parse_qs
import uuid
import zipfile
import numpy as np
import torch
from .artifacts import Artifact, validate_compatibility
from .evaluate import classification
from .merge import merge
from .repair import Capability, BuildBudget, repair

class Workbench:
    def __init__(self,project: Path):
        self.project=project.resolve()
        self.config=json.loads((self.project/"study/workbench.json").read_text())
        self.paths={entry["id"]:(self.project/entry["path"]).resolve() for entry in [self.config["base"],*self.config["experts"]]}
        if any(not p.is_relative_to(self.project) for p in self.paths.values()):
            raise ValueError("Catalog paths must stay in the project")
        self.lock=Lock();self.results={};self.archives={}
        with np.load(self.project/"study/workbench.npz",allow_pickle=False) as data:
            self.batches={c["name"]:(torch.from_numpy(data[f'x_{c["name"]}'].copy()).float(),torch.from_numpy(data[f'y_{c["name"]}'].copy()).long()) for c in self.config["capabilities"]}
            self.calibration={c["name"]:(torch.from_numpy(data[f'cal_x_{c["name"]}'].copy()).float(),torch.from_numpy(data[f'cal_y_{c["name"]}'].copy()).long()) for c in self.config["capabilities"]}
    def catalog(self):
        base=Artifact.open(self.paths[self.config["base"]["id"]]);experts=[]
        for info in self.config["experts"]:
            entry={**info,"compatible":False}
            try:validate_compatibility(base,[Artifact.open(self.paths[info["id"]])]);entry["compatible"]=True
            except ValueError as exc:entry["error"]=str(exc)
            experts.append(entry)
        return {"base":{**self.config["base"],"architecture":base.manifest["architecture"],"contract":base.manifest["contract"]},
                "experts":experts,"capabilities":self.config["capabilities"],"results":list(self.results.values()),"study":self.config.get("study",{})}
    def build(self,request):
        ids=request["experts"]
        allowed={i["id"] for i in self.config["experts"]}
        if not isinstance(ids,list) or not ids or len(set(ids))!=len(ids) or not set(ids)<=allowed:
            raise ValueError("Choose one or more unique catalog specialists")
        floors=request["floors"]
        names={c["name"] for c in self.config["capabilities"]}
        if set(floors)!=names or any(type(v) not in (int,float) or not 0<=v<=1 for v in floors.values()):
            raise ValueError("Every capability needs a floor in [0,1]")
        if type(request.get("repair",False)) is not bool:raise ValueError("repair must be boolean")
        requested_budget=request.get("budget",{})
        if not isinstance(requested_budget,dict) or not set(requested_budget)<={"rank","rounds"}:
            raise ValueError("Budget accepts rank and rounds")
        if any(type(v) is not int for v in requested_budget.values()):raise ValueError("Budget values must be integers")
        build_budget=BuildBudget(rank=requested_budget.get("rank",8),rounds=requested_budget.get("rounds",4),learning_rates=(.04,.12,.35))
        if not 0<=build_budget.rank<=64 or not 0<=build_budget.rounds<=32:raise ValueError("Rank must be 0..64 and rounds 0..32")
        key=uuid.uuid4().hex
        root=self.project/"exports"/key
        base=Artifact.open(self.paths[self.config["base"]["id"]])
        experts=[Artifact.open(self.paths[i]) for i in ids]
        mapping={i["id"]:i["task"] for i in self.config["experts"]}
        method=request.get("method","task_arithmetic")
        ordinary=merge(base,experts,root/"ordinary",method=method,scale=float(request.get("scale",.5)),
                       calibration=[self.calibration[mapping[i]][0] for i in ids] if method=="regmean" else None)
        initial={n:classification(ordinary.model(),*b) for n,b in self.batches.items()}
        report={"status":"ordinary merge exported","worst_deficit":max(max(0.,floors[n]-initial[n]["accuracy"]) for n in names),
                "fraction_floors_met":sum(initial[n]["accuracy"]>=floors[n] for n in names)/len(names),"rank_used_upper_bound":0,"attempts":0}
        final=ordinary
        if request.get("repair",False):
            caps=[Capability(c["name"],floors[c["name"]],c.get("protected",False),c.get("max_regression",.01)) for c in self.config["capabilities"]]
            final,report=repair(ordinary,self.calibration,self.batches,caps,root/"repaired",budget=build_budget)
        metrics={n:classification(final.model(),*b) for n,b in self.batches.items()}
        self.paths[key]=final.path
        archive=root/"modelchord.zip"
        with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
            for p in sorted(final.path.iterdir()):z.write(p,p.name)
            z.writestr("build-report.json",json.dumps(report,indent=2))
        self.archives[key]=archive
        result={"id":key,"label":f"{method} {'+ repair' if request.get('repair') else ''}","report":report,"status":report["status"],
                "metrics":metrics,"ordinary_metrics":initial,"download_url":f"/api/download/{key}",
                "model":{"bytes":sum(p.stat().st_size for p in final.path.iterdir()),"parameters":sum(t.numel() for t in final.state().values())},
                "recipe":final.manifest["provenance"]}
        (root/"result.json").write_text(json.dumps(result,indent=2))
        self.results[key]=result
        return result

def serve(project: Path,port: int = 8765):
    if not 1024<=port<=65535:raise ValueError("Choose an unprivileged port 1024..65535")
    workbench=Workbench(project)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def _send(self,value,status=200,content_type="application/json"):
            data=json.dumps(value,allow_nan=False).encode() if content_type=="application/json" else value
            self.send_response(status);self.send_header("Content-Type",content_type);self.send_header("Content-Length",str(len(data)))
            self.send_header("X-Content-Type-Options","nosniff");self.send_header("Cache-Control","no-store")
            self.send_header("Content-Security-Policy","default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(data)
        def _origin_ok(self):
            host=self.headers.get("Host","")
            if host not in {f"127.0.0.1:{port}",f"localhost:{port}"}:return False
            origin=self.headers.get("Origin")
            return origin is None or origin in {f"http://127.0.0.1:{port}",f"http://localhost:{port}"}
        def do_GET(self):
            if not self._origin_ok():return self._send({"error":"Loopback origin required"},403)
            route=urlparse(self.path)
            try:
                if route.path=="/api/catalog":return self._send(workbench.catalog())
                if route.path=="/api/sample":
                    name=parse_qs(route.query).get("domain",[workbench.config["capabilities"][0]["name"]])[0]
                    x,y=workbench.batches[name]
                    return self._send({"inputs":x[:1].tolist(),"label":str(int(y[0])),"domain":name})
                if route.path.startswith("/api/download/"):
                    key=route.path.rsplit("/",1)[1]
                    return self._send(workbench.archives[key].read_bytes(),content_type="application/zip")
                static={"/":"index.html","/index.html":"index.html","/style.css":"style.css","/app.js":"app.js"}
                if route.path not in static:return self._send({"error":"Not found"},404)
                name=static[route.path];mime={"html":"text/html; charset=utf-8","css":"text/css","js":"text/javascript"}[name.rsplit(".",1)[1]]
                return self._send(files("modelchord").joinpath("web",name).read_bytes(),content_type=mime)
            except (ValueError,KeyError,OSError) as exc:return self._send({"error":str(exc)},400)
        def do_POST(self):
            if not self._origin_ok():return self._send({"error":"Loopback origin required"},403)
            if self.headers.get("Content-Type","").split(";")[0]!="application/json":return self._send({"error":"JSON required"},415)
            try:
                size=int(self.headers.get("Content-Length","0"))
                if not 0<size<=65536:raise ValueError("Request size must be 1..65536 bytes")
                data=json.loads(self.rfile.read(size))
                if self.path=="/api/build":
                    if not workbench.lock.acquire(blocking=False):return self._send({"error":"Another build is running"},409)
                    try:result=workbench.build(data)
                    finally:workbench.lock.release()
                    return self._send(result)
                if self.path=="/api/predict":
                    artifact=Artifact.open(workbench.paths[data["artifact"]])
                    probs=artifact.predict(torch.tensor(data["inputs"],dtype=torch.float32)).softmax(1)
                    classes=artifact.manifest["architecture"]["classes"]
                    return self._send({"classes":classes,"labels":[classes[int(i)] for i in probs.argmax(1)],"probabilities":probs.tolist()})
                return self._send({"error":"Not found"},404)
            except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:return self._send({"error":str(exc)},400)
    print(f"MODELCHORD workbench: http://127.0.0.1:{port}",flush=True)
    ThreadingHTTPServer(("127.0.0.1",port),Handler).serve_forever()
