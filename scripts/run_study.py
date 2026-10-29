#!/usr/bin/env python3
"""Reproduce the bounded MODELCHORD study without downloaded model weights.

Training/calibration/selection complete before a confirmation lock is written.
The `--phase confirm` command refuses an absent lock and never tunes recipes.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"examples"))
from modelchord import Artifact,MLP,export_artifact,merge,repair,Capability,BuildBudget
from modelchord.artifacts import canonical,digest
from modelchord.evaluate import classification
from text_features import text_features,transform_text

TASKS=["clean","left","right","inverted"]
FLOORS={"clean":.88,"left":.80,"right":.80,"inverted":.80}
CAPS=[Capability(t,FLOORS[t],t=="clean",.02) for t in TASKS]
BUDGET=BuildBudget(rank=8,rounds=4,candidate_limit=72,learning_rates=(.04,.12,.35),ranks=(1,2))
torch.set_num_threads(4)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(canonical(value)+b"\n")


def load(path):return json.loads(Path(path).read_text())


def split_sources(x,y,seed):
    # Group exact source duplicates before splitting; retain all members together.
    groups={}
    for i,row in enumerate(x):
        key=hashlib.sha256(row.tobytes() if isinstance(row,np.ndarray) else row.encode()).hexdigest()
        groups.setdefault(key,[]).append(i)
    keys=sorted(groups)
    labels=np.array([y[groups[k][0]] for k in keys])
    train,remaining=train_test_split(np.arange(len(keys)),test_size=.45,random_state=seed,stratify=labels)
    cal,remaining=train_test_split(remaining,test_size=35/45,random_state=seed+1,stratify=labels[remaining])
    select,confirm=train_test_split(remaining,test_size=20/35,random_state=seed+2,stratify=labels[remaining])
    return {name:sorted(i for j in ids for i in groups[keys[j]]) for name,ids in zip(["train","calibration","selection","confirmation"],[train,cal,select,confirm])}


def vision_transform(x,task):
    a=x.reshape(-1,8,8).clone()
    if task=="left":
        a[:,:,:-1]=a[:,:,1:].clone();a[:,:,-1]=0
    elif task=="right":
        a[:,:,1:]=a[:,:,:-1].clone();a[:,:,0]=0
    elif task=="inverted":a=1-a
    elif task!="clean":raise ValueError(task)
    return a.reshape(-1,64)


def vision_data():
    d=load_digits();x=(d.data/16).astype(np.float32);y=d.target.astype(np.int64)
    return x,y


def partitions(x,y,split,tasks=TASKS,transform=vision_transform):
    # Caller requests calibration/selection explicitly; confirmation is constructed only after lock.
    return {p:{t:(transform(torch.tensor(x[ids]),t),torch.tensor(y[ids])) for t in tasks} for p,ids in split.items() if p!="confirmation"}


def train_full(model,x,y,epochs,seed,lr=.006):
    torch.manual_seed(seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad();loss=F.cross_entropy(model(x),y);loss.backward();optimizer.step()
    return model.eval()


def score(model,batches):return {n:classification(model,*b) for n,b in batches.items()}


def objective(metrics,floors=FLOORS):
    ds=[max(0.,floor-metrics[n]["accuracy"]) for n,floor in floors.items()]
    return (max(ds),sum(ds),sum(v["cross_entropy"] for v in metrics.values())/len(metrics))


def metrics_summary(metrics,floors=FLOORS,reference=None,protected=("clean",)):
    ds={n:max(0.,f-metrics[n]["accuracy"]) for n,f in floors.items()}
    return {"fraction_floors_met":sum(d<1e-8 for d in ds.values())/len(ds),"worst_deficit":max(ds.values()),"deficits":ds,
            "protected_regression":{n:(reference[n]["accuracy"]-metrics[n]["accuracy"]) if reference else 0. for n in protected}}


def latency(model,input_dim):
    x=torch.linspace(0,1,input_dim).reshape(1,-1)
    with torch.inference_mode():
        for _ in range(20):model(x)
        samples=[]
        for _ in range(100):
            start=time.perf_counter_ns();model(x);samples.append((time.perf_counter_ns()-start)/1e6)
    return {"batch":1,"warmup":20,"repeats":100,"median_ms":float(np.median(samples)),"p95_ms":float(np.percentile(samples,95))}


def provenance(suite,seed,split_hash,**extra):
    return {"suite":suite,"seed":seed,"split_sha256":split_hash,"protocol_sha256":digest(ROOT/"docs/PROTOCOL.md"),"pretraining":"none; trained locally from random initialization",**extra}


def export(path,model,contract,base_id=None,**p):
    return export_artifact(path,model,contract,base_id=base_id,provenance=p,license="MIT")


def baseline_fresh(artifact):
    # Actual independent invocation of an exported conventional artifact, before repair.
    code='''import json,sys,torch,socket\nfrom modelchord import Artifact\ndef denied(*a,**k):raise RuntimeError("network unavailable")\nsocket.socket=denied\na=Artifact.open(sys.argv[1]);d=a.manifest["architecture"]["input_dim"];x=torch.stack([torch.zeros(d),torch.ones(d)]);z=a.predict(x);assert not torch.equal(z[0],z[1]);print(json.dumps({"shape":list(z.shape),"changed_input_changed_output":True,"network":"socket disabled"}))'''
    with tempfile.TemporaryDirectory(prefix="modelchord-baseline-") as td:
        copy=Path(td)/"static";shutil.copytree(artifact.path,copy)
        result=subprocess.run([sys.executable,"-c",code,str(copy)],check=True,capture_output=True,text=True,cwd=td,env={"PATH":os.environ.get("PATH",""),"PYTHONPATH":str(ROOT/"src"),"HOME":td})
    return json.loads(result.stdout)


def train_vision(run_tag="", inverted_epochs=65, inverted_lr=.006):
    x,y=vision_data();split=split_sources(x,y,20261029)
    split_path=ROOT/"study/vision-split.json"
    source_hash=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
    split_record={"dataset":"sklearn.load_digits original UCI test subset","source_sha256":source_hash,"groups":"exact pixel duplicates grouped","indices":split,"sizes":{k:len(v) for k,v in split.items()},"confirmation_accessed":False}
    write(split_path,split_record);split_hash=digest(split_path)
    data=partitions(x,y,split)
    contract={"family":"vision","representation":"digits-8x8-float32/v1","preprocessing":"flatten row-major; divide original [0,16] pixels by 16; no resize; shifts/inversion are task domains","labels":[str(i) for i in range(10)],"tokenizer":None}
    registry=load(ROOT/"study/recipe-registry.json") if run_tag and (ROOT/"study/recipe-registry.json").exists() else [];selected_scales={};study_seeds=[]
    for seed in [11,29]:
        start=time.perf_counter();directory=ROOT/f"artifacts/vision-seed{seed}{run_tag}";directory.mkdir(parents=True,exist_ok=True)
        torch.manual_seed(seed);base_model=MLP(64,[96,64],contract["labels"])
        base_model=train_full(base_model,*data["train"]["clean"],epochs=160,seed=seed)
        base=export(directory/"base",base_model,contract,**provenance("vision",seed,split_hash,training={"epochs":160,"optimizer":"Adam","lr":.006,"examples_per_epoch":len(split["train"])},data_sha256=source_hash))
        experts=[];training={"base_seconds":time.perf_counter()-start}
        for task in TASKS[1:]:
            expert_start=time.perf_counter();expert_epochs=inverted_epochs if task=="inverted" else 65;expert_lr=inverted_lr if task=="inverted" else .006;model=train_full(deepcopy(base_model),*data["train"][task],epochs=expert_epochs,seed=seed+100,lr=expert_lr)
            artifact=export(directory/f"expert-{task}",model,contract,base.manifest["base_id"],**provenance("vision",seed,split_hash,task=task,training={"epochs":expert_epochs,"optimizer":"Adam","lr":expert_lr,"examples_per_epoch":len(split["train"])},data_sha256=source_hash))
            experts.append(artifact);training[f"expert_{task}_seconds"]=time.perf_counter()-expert_start
        if seed==11:
            np.savez(ROOT/"study/workbench.npz",**{f"x_{t}":data["selection"][t][0].numpy() for t in TASKS},**{f"y_{t}":data["selection"][t][1].numpy() for t in TASKS},**{f"cal_x_{t}":data["calibration"][t][0].numpy() for t in TASKS},**{f"cal_y_{t}":data["calibration"][t][1].numpy() for t in TASKS})
            write(ROOT/"study/workbench.json",{"base":{"id":"base","path":str(base.path.relative_to(ROOT))},"experts":[{"id":t,"label":f"Digit {t} specialist","task":t,"path":str(e.path.relative_to(ROOT))} for t,e in zip(TASKS[1:],experts)],"capabilities":[{"name":c.name,"floor":c.floor,"protected":c.protected,"max_regression":c.max_regression} for c in CAPS],"study":{"protocol":"docs/PROTOCOL.md","suite":"UCI 8x8 digit domains; calibration and selection only"}})
        base_selection=score(base_model,data["selection"])
        expert_evidence={task:{"base":base_selection[task],"expert":classification(expert.model(),*data["selection"][task]),"improves":classification(expert.model(),*data["selection"][task])["accuracy"]>base_selection[task]["accuracy"]} for task,expert in zip(TASKS[1:],experts)}
        if not all(e["improves"] for e in expert_evidence.values()):
            raise RuntimeError(f"Meaningless expert baseline: {expert_evidence}")
        rows=[{"name":"base","artifact":str(base.path.relative_to(ROOT))}]+[{"name":f"expert-{t}","artifact":str(e.path.relative_to(ROOT))} for t,e in zip(TASKS[1:],experts)]
        selected={}
        for method in ["task_arithmetic","ties","dare","tsv","regmean"]:
            candidates=[]
            settings=[(a,d) for a in [.25,.5,.75,1.] for d in ([.2,.5] if method in ["ties","dare"] else [.2])]
            if method=="regmean":settings=[(1.,.2)]
            if seed==29:settings=[selected_scales[method]]
            for index,(scale,density) in enumerate(settings):
                t0=time.perf_counter();artifact=merge(base,experts,directory/f"{method}-{index}",method=method,scale=scale,density=density,seed=20261029,calibration=[data["calibration"][t][0] for t in TASKS[1:]] if method=="regmean" else None)
                s=score(artifact.model(),data["selection"])
                row={"suite":"vision","seed":seed,"name":method,"artifact":str(artifact.path.relative_to(ROOT)),"parameters":{"scale":scale,"density":density},"selection":s,"selection_objective":list(objective(s)),"build_seconds":time.perf_counter()-t0,"data_assisted":method=="regmean","gradient_examples":0,"teacher_calls":0,"selection_examples":sum(len(b[1]) for b in data["selection"].values())}
                registry.append(row);candidates.append(row)
            winner=min(candidates,key=lambda r:tuple(r["selection_objective"]))
            selected[method]=winner
            if seed==11:selected_scales[method]=(winner["parameters"]["scale"],winner["parameters"]["density"])
            rows.append(dict(winner))
        best_baseline=min(selected.values(),key=lambda r:tuple(r["selection_objective"]))
        ordinary=Artifact.open(ROOT/selected["task_arithmetic"]["artifact"])
        independent=baseline_fresh(ordinary)
        t0=time.perf_counter();repaired,repair_report=repair(ordinary,data["calibration"],data["selection"],CAPS,directory/"repaired",budget=BUDGET)
        repair_seconds=time.perf_counter()-t0
        write(ROOT/f"study/vision-seed{seed}-repair.json",repair_report)
        rows.append({"name":"repair","artifact":str(repaired.path.relative_to(ROOT)),"build_seconds":repair_seconds,"gradient_examples":repair_report["gradient_examples"],"teacher_calls":0,"selection_examples":repair_report["validation_examples"],"rank_used":repair_report["rank_used_upper_bound"],"selection":repair_report["final"]})
        registry.append({"suite":"vision","seed":seed,"name":"repair","artifact":str(repaired.path.relative_to(ROOT)),"budget":repair_report["budget"],"status":repair_report["status"],"data_assisted":True})
        if best_baseline["artifact"]!=str(ordinary.path.relative_to(ROOT)):
            strongest=Artifact.open(ROOT/best_baseline["artifact"]);tbest=time.perf_counter()
            a,rr=repair(strongest,data["calibration"],data["selection"],CAPS,directory/"repair-strongest",budget=BUDGET)
            write(ROOT/f"study/vision-seed{seed}-strongest-repair.json",rr)
            rows.append({"name":"repair-strongest","ordinary_method":best_baseline["name"],"artifact":str(a.path.relative_to(ROOT)),"build_seconds":time.perf_counter()-tbest,"gradient_examples":rr["gradient_examples"],"teacher_calls":0,"selection_examples":rr["validation_examples"],"rank_used":rr["rank_used_upper_bound"],"selection":rr["final"],"selection_objective":list(objective(rr["final"]))})
            registry.append({"suite":"vision","seed":seed,"name":"repair-strongest","ordinary_method":best_baseline["name"],"budget":rr["budget"],"status":rr["status"]})

        # Exactly match the calibration gradient-access sequence and count.
        round_targets={entry["round"]:entry["target"] for entry in repair_report["trace"]}
        joint_rows=[]
        for lr in [.002,.006,.02]:
            model=ordinary.model();optimizer=torch.optim.SGD(model.parameters(),lr=lr);consumed=0;t0=time.perf_counter()
            for _,target in sorted(round_targets.items()):
                xx,yy=data["calibration"][target];optimizer.zero_grad();F.cross_entropy(model(xx),yy).backward();optimizer.step();consumed+=len(yy)
            if consumed!=repair_report["gradient_examples"]:raise RuntimeError("Joint calibration budget mismatch")
            artifact=export(directory/f"joint-{lr}",model,contract,base.manifest["base_id"],**provenance("vision",seed,split_hash,operation="ordinary joint fine-tuning",lr=lr,gradient_steps=len(round_targets),gradient_examples=consumed,teacher_calls=0))
            s=score(model,data["selection"])
            r={"name":"target_matched_finetune","artifact":str(artifact.path.relative_to(ROOT)),"selection":s,"selection_objective":list(objective(s)),"gradient_examples":consumed,"gradient_steps":len(round_targets),"teacher_calls":0,"lr":lr,"build_seconds":time.perf_counter()-t0,"selection_examples":sum(len(b[1]) for b in data["selection"].values()),"matching":"exact repair gradient steps/examples/teacher calls; distinct candidate evaluation cost"}
            joint_rows.append(r);registry.append({"suite":"vision","seed":seed,**r})
        rows.append(min(joint_rows,key=lambda r:tuple(r["selection_objective"])))
        joint_mixture_rows=[]
        pool_x=torch.cat([data["calibration"][t][0] for t in TASKS]);pool_y=torch.cat([data["calibration"][t][1] for t in TASKS])
        for lr in [.002,.006,.02]:
            model=ordinary.model();optimizer=torch.optim.SGD(model.parameters(),lr=lr);consumed=0;t0=time.perf_counter();generator=torch.Generator().manual_seed(8801)
            for _,target in sorted(round_targets.items()):
                count=len(data["calibration"][target][1]);indices=torch.randperm(len(pool_y),generator=generator)[:count]
                optimizer.zero_grad();F.cross_entropy(model(pool_x[indices]),pool_y[indices]).backward();optimizer.step();consumed+=count
            if consumed!=repair_report["gradient_examples"]:raise RuntimeError("Joint mixture budget mismatch")
            artifact=export(directory/f"joint-mixture-{lr}",model,contract,base.manifest["base_id"],**provenance("vision",seed,split_hash,operation="ordinary joint mixture fine-tuning",lr=lr,gradient_steps=len(round_targets),gradient_examples=consumed,teacher_calls=0))
            ss=score(model,data["selection"])
            row={"name":"joint_finetune","artifact":str(artifact.path.relative_to(ROOT)),"selection":ss,"selection_objective":list(objective(ss)),"gradient_examples":consumed,"gradient_steps":len(round_targets),"teacher_calls":0,"lr":lr,"build_seconds":time.perf_counter()-t0,"selection_examples":sum(len(b[1]) for b in data["selection"].values()),"matching":"exact steps and total example accesses; balanced-source domain pool; distinct candidate evaluation cost"}
            joint_mixture_rows.append(row);registry.append({"suite":"vision","seed":seed,**row})
        rows.append(min(joint_mixture_rows,key=lambda r:tuple(r["selection_objective"])))

        if seed==11:
            # Required controls and ablations are all registered before confirmation.
            for label,options in [("no_priority",{"prioritize":False}),("fixed_rank",{"allocate":False}),("no_protection",{"protect":False}),("no_correction",{"correction":False})]:
                t0=time.perf_counter();a,r=repair(ordinary,data["calibration"],data["selection"],CAPS,directory/f"ablation-{label}",budget=BUDGET,**options)
                write(ROOT/f"study/ablation-{label}.json",r);rows.append({"name":f"ablation-{label}","artifact":str(a.path.relative_to(ROOT)),"build_seconds":time.perf_counter()-t0,"gradient_examples":r["gradient_examples"],"selection_examples":r["validation_examples"],"rank_used":r["rank_used_upper_bound"],"selection":r["final"]})
                registry.append({"suite":"vision","seed":seed,"name":f"ablation-{label}","options":options,"status":r["status"]})
            tiny={t:(xx[:1],yy[:1]) for t,(xx,yy) in data["calibration"].items()}
            a,r=repair(ordinary,tiny,data["selection"],CAPS,directory/"insufficient-calibration",budget=BUDGET)
            write(ROOT/"study/insufficient-calibration.json",r);rows.append({"name":"insufficient-calibration","artifact":str(a.path.relative_to(ROOT)),"selection":r["final"],"rank_used":r["rank_used_upper_bound"],"gradient_examples":r["gradient_examples"],"selection_examples":r["validation_examples"]})
            easy=merge(base,[experts[0]]*3,directory/"easy-ordinary",method="average",scale=1.)
            easy_metrics=score(easy.model(),data["selection"])
            easy_caps=[Capability(t,max(0.,easy_metrics[t]["accuracy"]-.03)) for t in TASKS]
            a,r=repair(easy,data["calibration"],data["selection"],easy_caps,directory/"easy-repaired",budget=BUDGET)
            write(ROOT/"study/easy-merge.json",r);rows.append({"name":"easy-merge","artifact":str(a.path.relative_to(ROOT)),"custom_floors":{c.name:c.floor for c in easy_caps},"selection":r["final"],"rank_used":r["rank_used_upper_bound"]})
            bad=train_full(deepcopy(base_model),data["train"]["inverted"][0],(data["train"]["inverted"][1]+1)%10,epochs=65,seed=seed+200)
            conflict=export(directory/"expert-conflict",bad,contract,base.manifest["base_id"],**provenance("vision",seed,split_hash,conflict="intentionally rotated training labels +1 modulo10; declared output labels remain digit identities"))
            c=merge(base,[experts[0],experts[1],conflict],directory/"conflict-ordinary",method="task_arithmetic",scale=selected["task_arithmetic"]["parameters"]["scale"])
            a,r=repair(c,data["calibration"],data["selection"],CAPS,directory/"conflict-repaired",budget=BUDGET)
            write(ROOT/"study/conflict.json",r)
            rows.extend([{"name":"conflict-ordinary","artifact":str(c.path.relative_to(ROOT))},{"name":"conflict-repaired","artifact":str(a.path.relative_to(ROOT)),"selection":r["final"],"rank_used":r["rank_used_upper_bound"]}])
            registry.extend({"suite":"vision","seed":seed,"name":name,"registered":True} for name in ["insufficient-calibration","easy-merge","conflict-ordinary","conflict-repaired"])
        study_seeds.append({"seed":seed,"training":training,"expert_selection_evidence":expert_evidence,"independent_ordinary":independent,"rows":rows,"ordinary":selected["task_arithmetic"]["artifact"],"experts":[str(e.path.relative_to(ROOT)) for e in experts],"training_build_total_seconds":time.perf_counter()-start})
        write(ROOT/"study/vision-pending.json",{"split_sha256":split_hash,"seeds":study_seeds,"fixed_baseline_parameters":selected_scales,"floors":FLOORS})
        write(ROOT/"study/recipe-registry.json",registry)
        print(f"vision seed {seed} trained; expert improvement {expert_evidence}; repair {repair_report['status']} rank={repair_report['rank_used_upper_bound']}",flush=True)
    return registry


class AdaptedLinear(nn.Module):
    def __init__(self,base,rank=4):
        super().__init__();self.base=deepcopy(base)
        for p in self.base.parameters():p.requires_grad=False
        self.a=nn.Parameter(torch.randn(rank,base.in_features)*.02);self.b=nn.Parameter(torch.zeros(base.out_features,rank))
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.a),self.b)
    def folded(self):
        m=deepcopy(self.base)
        with torch.no_grad():m.weight.add_(self.b@self.a)
        return m


def sms_data():
    path=ROOT/"study/private/sms.zip"
    if not path.exists():raise FileNotFoundError("Download official CC BY4.0 UCI SMS zip to study/private/sms.zip first")
    with zipfile.ZipFile(path) as z:
        raw=z.read("SMSSpamCollection").decode("utf-8")
    records={}
    for line in raw.splitlines():
        if "\t" not in line:continue
        label,text=line.split("\t",1)
        if label not in ["ham","spam"]:continue
        if text in records and records[text]!=label:raise ValueError("Conflicting duplicate labels")
        records[text]=label
    classes={label:sorted(t for t,l in records.items() if l==label) for label in ["ham","spam"]}
    rng=np.random.default_rng(20261030);n=min(map(len,classes.values()))
    texts=[];ys=[]
    for label,values in classes.items():
        selected=rng.choice(values,n,replace=False).tolist();texts.extend(selected);ys.extend([0 if label=="ham" else 1]*n)
    return texts,np.array(ys,dtype=np.int64),digest(path),len(records)


def text_batches(texts,y,split,parts):
    tasks=["clean","uppercase","leet","underscores"]
    return {p:{t:(torch.tensor(text_features(transform_text([texts[i] for i in split[p]],t))),torch.tensor(y[split[p]])) for t in tasks} for p in parts}


def train_text(registry):
    texts,y,source_hash,unique=sms_data();split=split_sources(texts,y,20261030)
    record={"dataset":"UCI SMS Spam Collection","zip_sha256":source_hash,"processed_sha256":hashlib.sha256(canonical({"texts":texts,"labels":y.tolist()})).hexdigest(),"unique_records":unique,"balanced_examples":len(texts),"ids":{p:[hashlib.sha256(texts[i].encode()).hexdigest() for i in ids] for p,ids in split.items()},"sizes":{p:len(ids) for p,ids in split.items()},"confirmation_accessed":False}
    write(ROOT/"study/text-split.json",record);split_hash=digest(ROOT/"study/text-split.json")
    data=text_batches(texts,y,split,["train","calibration","selection"])
    tasks=list(data["train"]);floors={t:.75 if t=="clean" else .72 for t in tasks};caps=[Capability(t,floors[t],t=="clean",.03) for t in tasks]
    contract={"family":"text","representation":"utf8-signed-blake2b-unigram-bigram-256-l2/v1","preprocessing":"case-sensitive UTF-8 bytes; signed 256-bin unigram+bigram counts; L2 normalize; BLAKE2b person MChordV1; see examples/text_features.py","labels":["ham","spam"],"tokenizer":{"kind":"fixed UTF-8 byte unigrams+bigrams","vocab_fit":False,"dimension":256,"max_characters":10000}}
    directory=ROOT/"artifacts/text";directory.mkdir(parents=True,exist_ok=True);t0=time.perf_counter();torch.manual_seed(43)
    model=train_full(MLP(256,[64,32],contract["labels"]),*data["train"]["clean"],epochs=160,seed=43,lr=.004)
    base=export(directory/"base",model,contract,**provenance("text",43,split_hash,data_sha256=source_hash,training={"epochs":160,"optimizer":"Adam","lr":.004,"examples_per_epoch":len(split["train"])},source="UCI SMS Spam Collection; deduplicated class-balanced subset"))
    experts=[];rows=[{"name":"base","artifact":str(base.path.relative_to(ROOT))}];expert_evidence={}
    for task in tasks[1:]:
        adapted=deepcopy(model);adapted.layers=nn.ModuleList(AdaptedLinear(layer,4) for layer in adapted.layers)
        train_full(adapted,*data["train"][task],epochs=100,seed=44,lr=.008)
        folded=deepcopy(model);folded.layers=nn.ModuleList(layer.folded() for layer in adapted.layers)
        expert=export(directory/f"expert-{task}",folded,contract,base.manifest["base_id"],**provenance("text",43,split_hash,task=task,adapter={"rank":4,"trainable":"A/B factors only; all base weights and biases frozen","epochs":100,"optimizer":"Adam","lr":.008},data_sha256=source_hash))
        from safetensors.torch import save_file
        state={f"layers.{i}.{part}":getattr(layer,part).detach().contiguous() for i,layer in enumerate(adapted.layers) for part in ["a","b"]}
        adapter_file=directory/f"adapter-{task}.safetensors";save_file(state,adapter_file)
        write(directory/f"adapter-{task}.json",{"schema":"modelchord/lora-v1","base_weights_sha256":base.manifest["weights_sha256"],"base_id":base.manifest["base_id"],"rank":4,"scale":1.,"adapter_sha256":digest(adapter_file),"folded_expert_weights_sha256":expert.manifest["weights_sha256"],"contract":contract})
        experts.append(expert);rows.append({"name":f"expert-{task}","artifact":str(expert.path.relative_to(ROOT))})
        eb=classification(base.model(),*data["selection"][task]);ee=classification(expert.model(),*data["selection"][task]);expert_evidence[task]={"base":eb,"expert":ee,"improves":ee["accuracy"]>eb["accuracy"]}
    candidates=[]
    for i,scale in enumerate([.25,.5,.75,1.]):
        ordinary=merge(base,experts,directory/f"ordinary-{i}",method="task_arithmetic",scale=scale)
        scores=score(ordinary.model(),data["selection"]);r={"name":"task_arithmetic","artifact":str(ordinary.path.relative_to(ROOT)),"scale":scale,"selection":scores,"selection_objective":list(objective(scores,floors))};candidates.append(r);registry.append({"suite":"text",**r})
    best=min(candidates,key=lambda r:tuple(r["selection_objective"]));ordinary=Artifact.open(ROOT/best["artifact"]);rows.append(best)
    independent=baseline_fresh(ordinary)
    repaired,r=repair(ordinary,data["calibration"],data["selection"],caps,directory/"repaired",budget=BUDGET)
    write(ROOT/"study/text-repair.json",r);rows.append({"name":"repair","artifact":str(repaired.path.relative_to(ROOT)),"selection":r["final"],"rank_used":r["rank_used_upper_bound"],"gradient_examples":r["gradient_examples"]});registry.append({"suite":"text","name":"repair","budget":r["budget"],"status":r["status"]})
    write(ROOT/"study/text-pending.json",{"split_sha256":split_hash,"floors":floors,"rows":rows,"expert_selection_evidence":expert_evidence,"independent_ordinary":independent,"training_build_total_seconds":time.perf_counter()-t0,"adapter_total_bytes":sum(p.stat().st_size for p in directory.glob("adapter-*"))})
    write(ROOT/"study/recipe-registry.json",registry)
    print(f"text trained: expert improvements {expert_evidence}; repair {r['status']}",flush=True)


def freeze():
    if (ROOT/"study/confirmation-lock.json").exists():raise FileExistsError("Confirmation lock already exists; preserve it")
    files=[ROOT/"examples/text_features.py",ROOT/"study/deployment-selection.json",ROOT/"study/build-resource-measurement.json",ROOT/"study/text-adapter-audit.json",ROOT/"docs/PROTOCOL.md",ROOT/"study/vision-split.json",ROOT/"study/text-split.json",ROOT/"study/vision-pending.json",ROOT/"study/text-pending.json",ROOT/"study/recipe-registry.json",Path(__file__)]
    files += [ROOT/"src/modelchord"/f"{name}.py" for name in ["__init__","artifacts","models","merge","repair","evaluate"]]
    files += sorted((ROOT/"scripts").glob("data*.py"))
    files += sorted(p for p in (ROOT/"study").rglob("*.md") if p.name.lower().startswith("protocol"))
    files += sorted((ROOT/"study/pilot-v1").glob("*.json"))
    files += sorted((ROOT/"artifacts").rglob("manifest.json"))
    files += sorted((ROOT/"artifacts/text").glob("adapter-*.json")) + sorted((ROOT/"artifacts/text").glob("adapter-*.safetensors"))
    lock={"schema":"modelchord/confirmation-lock-v1","created_unix":time.time(),"confirmation_results_seen":False,"files":{str(p.relative_to(ROOT)):digest(p) for p in files},"artifact_weights":{str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/"artifacts").rglob("model.safetensors"))},"note":"All recipes locked before any confirmation metric is computed. Source changes after lock invalidate this study and require a new confirmation partition."}
    write(ROOT/"study/confirmation-lock.json",lock)


def confirm():
    lock=load(ROOT/"study/confirmation-lock.json")
    for relative,sha in {**lock["files"],**lock["artifact_weights"]}.items():
        if digest(ROOT/relative)!=sha:raise RuntimeError(f"Confirmation lock violated: {relative}")
    if (ROOT/"study/results.json").exists():raise FileExistsError("Confirmation already seen; preserve result and register a new study before revising")
    x,y=vision_data();vision_ledger=load(ROOT/"study/vision-split.json");split=vision_ledger["indices"];ids=split["confirmation"]
    if hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()!=vision_ledger["source_sha256"]:raise RuntimeError("Vision source content changed")
    if split_sources(x,y,20261029)!=split:raise RuntimeError("Vision partition membership changed")
    texts,yy,sms_zip_hash,_=sms_data();text_ledger=load(ROOT/"study/text-split.json");text_split=split_sources(texts,yy,20261030)
    if sms_zip_hash!=text_ledger["zip_sha256"]:raise RuntimeError("SMS source zip changed")
    if hashlib.sha256(canonical({"texts":texts,"labels":yy.tolist()})).hexdigest()!=text_ledger["processed_sha256"]:raise RuntimeError("Processed SMS content changed")
    current_ids={p:[hashlib.sha256(texts[i].encode()).hexdigest() for i in source_ids] for p,source_ids in text_split.items()}
    if current_ids!=text_ledger["ids"]:raise RuntimeError("SMS partition membership changed")
    if (ROOT/"study/confirmation-started.json").exists():raise FileExistsError("Confirmation was already started; preserve results and register a new confirmation set for revisions")
    write(ROOT/"study/confirmation-started.json",{"lock_sha256":digest(ROOT/"study/confirmation-lock.json"),"started_unix":time.time(),"note":"Written after all source/partition checks and before any confirmation metric."})
    batches={t:(vision_transform(torch.tensor(x[ids]),t),torch.tensor(y[ids])) for t in TASKS}
    pending=load(ROOT/"study/vision-pending.json");result={"schema":"modelchord/study-v1","protocol_sha256":digest(ROOT/"docs/PROTOCOL.md"),"confirmation_lock_sha256":digest(ROOT/"study/confirmation-lock.json"),"vision":[],"text":{},"peak_rss_bytes":0,"acceptable_clean_base_regression":0.10,"platform":"Apple M4 CPU; torch threads=4; macOS getrusage ru_maxrss in bytes","limitations":["Small same-source controlled digit domains; no natural multi-dataset or foundation-model claim","Text integration is a character-feature classifier with low-rank adapters, not an LLM","Held-out second expert initialization, shared data split; two seeds do not support significance claims","Source digit subset came from original UCI test file; no external pretraining or benchmark contamination claim","Repair uses labeled calibration and repeated selection; not training-free","Joint baseline matches gradient examples/steps/teacher calls, not exact FLOPs or candidate selection cost","Confirmation accessed once after lock; negative results are retained"]}
    for seed in pending["seeds"]:
        ordinary=Artifact.open(ROOT/seed["ordinary"]);reference=score(ordinary.model(),batches);evaluated=[]
        base_clean=classification(Artifact.open(ROOT/next(r["artifact"] for r in seed["rows"] if r["name"]=="base")).model(),*batches["clean"])["accuracy"]
        for row in seed["rows"]:
            a=Artifact.open(ROOT/row["artifact"]);model=a.model();metrics=score(model,batches)
            row={**row,"confirmation":metrics,"summary":metrics_summary(metrics,row.get("custom_floors",FLOORS),reference),"clean_base_regression":base_clean-metrics["clean"]["accuracy"],"clean_base_regression_bound_met":base_clean-metrics["clean"]["accuracy"]<=.10+1e-9,"stored_artifact_bytes":sum(p.stat().st_size for p in a.path.iterdir()),"weights_bytes":(a.path/"model.safetensors").stat().st_size,"latency":latency(model,64)}
            if row["name"]=="repair-strongest":
                parent_row=next(r for r in seed["rows"] if r["name"]==row["ordinary_method"])
                parent_clean=classification(Artifact.open(ROOT/parent_row["artifact"]).model(),*batches["clean"])["accuracy"]
                row["protected_parent_clean_regression"]=parent_clean-metrics["clean"]["accuracy"]
            row["summary"]["protected_reference"]="selected task arithmetic"
            evaluated.append(row)
        experts=[Artifact.open(ROOT/p) for p in seed["experts"]];models=[e.model() for e in experts]
        class Ensemble(nn.Module):
            def forward(self,x):return torch.stack([m(x).softmax(1) for m in models]).mean(0).clamp_min(1e-12).log()
        ensemble=Ensemble();metrics=score(ensemble,batches)
        evaluated.append({"name":"ensemble-reference","confirmation":metrics,"summary":metrics_summary(metrics,FLOORS,reference),"stored_artifact_bytes":sum(sum(p.stat().st_size for p in e.path.iterdir()) for e in experts),"weights_bytes":sum((e.path/"model.safetensors").stat().st_size for e in experts),"latency":latency(ensemble,64),"note":"Retains all three experts; quality/cost reference, not equivalent-memory baseline"})
        result["vision"].append({**{k:v for k,v in seed.items() if k!="rows"},"floors":FLOORS,"confirmation_source_count":len(ids),"rows":evaluated})
    tb=text_batches(texts,yy,text_split,["confirmation"])["confirmation"]
    tp=load(ROOT/"study/text-pending.json");ordinary=next(r for r in tp["rows"] if r["name"]=="task_arithmetic");reference=score(Artifact.open(ROOT/ordinary["artifact"]).model(),tb);rows=[]
    text_base_clean=classification(Artifact.open(ROOT/next(r["artifact"] for r in tp["rows"] if r["name"]=="base")).model(),*tb["clean"])["accuracy"]
    for row in tp["rows"]:
        a=Artifact.open(ROOT/row["artifact"]);m=a.model();metrics=score(m,tb);rows.append({**row,"confirmation":metrics,"summary":metrics_summary(metrics,tp["floors"],reference),"clean_base_regression":text_base_clean-metrics["clean"]["accuracy"],"stored_artifact_bytes":sum(p.stat().st_size for p in a.path.iterdir()),"weights_bytes":(a.path/"model.safetensors").stat().st_size,"latency":latency(m,256)})
    result["text"]={**{k:v for k,v in tp.items() if k!="rows"},"rows":rows,"confirmation_source_count":len(text_split["confirmation"])}
    result["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["total_artifact_bytes"]=sum(p.stat().st_size for p in (ROOT/"artifacts").rglob("*") if p.is_file())
    result["recipe_count"]=len(load(ROOT/"study/recipe-registry.json"))
    result["expert_diagnostics"]=load(ROOT/"study/inversion-training-diagnostic.json") if (ROOT/"study/inversion-training-diagnostic.json").exists() else []
    write(ROOT/"study/results.json",result)
    # Public changed-input examples are UCI digits (CC BY4.0), no raw SMS strings redistributed.
    sample_ids=ids[:12];np.savez(ROOT/"study/digit-examples.npz",inputs=x[sample_ids],labels=y[sample_ids],source_ids=np.array(sample_ids))
    print(json.dumps({"vision":[{"seed":s["seed"],"rows":[{"name":r["name"],"accuracy":{t:round(m["accuracy"],4) for t,m in r["confirmation"].items()},"summary":r["summary"]} for r in s["rows"]]} for s in result["vision"]],"text":[{"name":r["name"],"summary":r["summary"]} for r in result["text"]["rows"]]},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=["train","confirm","all"],default="all");args=parser.parse_args()
    (ROOT/"study/private").mkdir(parents=True,exist_ok=True)
    if args.phase in ["train","all"]:
        if (ROOT/"study/confirmation-lock.json").exists():raise FileExistsError("Existing study locked; use a new output workspace for a new study")
        registry=train_vision(run_tag="-v2",inverted_epochs=1200,inverted_lr=.001);train_text(registry)
        subprocess.run([sys.executable,str(ROOT/"scripts/data_adapter_audit.py")],check=True)
        subprocess.run([sys.executable,str(ROOT/"scripts/data_finalize.py")],check=True)
        freeze()
    if args.phase in ["confirm","all"]:confirm()

if __name__=="__main__":main()
