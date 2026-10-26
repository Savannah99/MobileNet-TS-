"""Fixed experiment definitions for joint selection on three datasets/two budgets."""
from pathlib import Path
import sys
import json
import datetime
import dataclasses
import shutil
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"code/conference/core9"))
from common import dataset, GRU, indices, window_labels, confusion, metric, save_json, sha256
from vision import Scorer, load_encoder, encoder_image

RUN = ROOT/"outputs/joint9_20261022"
DATASETS = ("cholec80", "autolaparo", "pedi38")
CONFIGS = {"dense16": 16, "ts8": 8, "ts4": 4}
BETA = .25
PYTHON = "/home/luning/miniconda3/envs/medai/bin/python"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


def ds_for(name):
    ds = dataset(name)
    if name == "autolaparo":
        ds = dataclasses.replace(ds, frames=encoder_image(ds, ds.ids("val")[0], 0).parent.parent,
                                 scores=ROOT/"outputs/autolaparo_unified_20261021/scores/adapt_e07")
    return ds


def scorer_path(name):
    return {
        "cholec80": RUN/"cholec_head/scorer.pt",
        "autolaparo": ROOT/"outputs/autolaparo_unified_20261021/checkpoints/adapt_e07.pt",
        "pedi38": ROOT/"checkpoints/pedi38_extension/mobilenet_ts_plus_pedi38.pt",
    }[name]


def scorer_for(name, headless=False):
    path = (ROOT/"checkpoints/pedi38_extension/mobilenet_ts_plus_cholec80_backbone.pt"
            if headless else scorer_path(name))
    obj = torch.load(path, map_location="cpu", weights_only=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = Scorer(ds_for(name).classes)
    for key in ("features", "trunk", "cls_head", "flow_head"):
        if key == "cls_head" and headless:
            assert key not in obj
            continue
        getattr(model, key).load_state_dict(obj[key])
    model.timing_proxy_head = False
    return model.cuda().eval()


def model_rows(name, config):
    rows = [r for r in read(RUN/"model_manifest.json")["records"]
            if (r["dataset"], r["config"]) == (name, config)]
    assert [r["seed"] for r in rows] == [0, 1, 2]
    for r in rows:
        assert sha256(r["checkpoint"]) == r["checkpoint_sha256"]
    return rows


def model_for(row):
    obj = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
    model = GRU(ds_for(row["dataset"]).classes).cuda()
    model.load_state_dict(obj.get("best_state", obj.get("best_single")))
    return model.eval()


def setup():
    configure()
    RUN.mkdir(exist_ok=True)
    if not (RUN/"model_manifest.json").exists():
        data = read(ROOT/"outputs/report22/data.json")
        records = []
        for d in DATASETS:
            for c in CONFIGS:
                rows = sorted([r for r in data["precision"] if (r["dataset"],r["config"])==(d,c)],
                              key=lambda r:r["seed"])
                for r in rows:
                    records.append({"dataset":d,"config":c,"seed":r["seed"],"k":CONFIGS[c],
                                    "checkpoint":r["checkpoint"],"checkpoint_sha256":sha256(r["checkpoint"]),
                                    "prior_window_j":r["window"]["jaccard"]})
        assert len(records)==27
        save_json(RUN/"model_manifest.json",{"records":records,"frozen_utc":now()})
        shutil.copy2(ROOT/"report22.md",RUN/"report22_before.md")
        shutil.copy2(ROOT/"outputs/report22/data.json",RUN/"report22_before_data.json")
        shutil.copy2(ROOT/"code/conference/report22/build_report22.py",RUN/"build_report22_before.py")
    protocol = {
        "datasets":list(DATASETS),"budgets":[8,4],"window":16,"stride":16,"fps":1,
        "beta":BETA,"beta_policy":"Fixed0.25 for all six joint configurations, transferred from completed Pedi38 study. No per-dataset or per-budget tuning. beta0 matched confidence controls are reported even if better.",
        "models_sha256":sha256(RUN/"model_manifest.json"),
        "models":"Freeze exact27 downstream models used by report22. No downstream retraining. Pedi current Dense and mixed_distill TS8; others original core versions. Same models for pure confidence and joint selection.",
        "selector":"One frame per ordered bin; first bin confidence; subsequent confidence+beta*clip(1-cosine64DMobileNettrunk_to_previous_selected,0,1). NumPyfloat64 normalization; context resets each16framewindow. k4 has4frames/bin,k8 has2.",
        "cholec_reconstruction":{"reason":"Saved scorer lacks cls_head; random head cannot supply phase confidence for new selection.",
            "source":str(ROOT/"checkpoints/pedi38_extension/mobilenet_ts_plus_cholec80_backbone.pt"),
            "source_sha256":sha256(ROOT/"checkpoints/pedi38_extension/mobilenet_ts_plus_cholec80_backbone.pt"),
            "recipe":"Freeze features,trunk,flow_head and BN. Train only new Linear64:7 using all training-frame labels and extracted64Dfeatures; Adam lr1e-3 wd1e-5,30epochs,batch4096,seed0,median-frequency weightedCE. Select earliest maxvalidation frame macroJ among epoch0..30, then export complete scorer. No test used for head training/selection.",
            "identity":"New reconstructed scorer,not reproduction of missing original classifier. Re-evaluate confidence-only controls with same new scorer."},
        "splits":"Original Chole40/8/32,Auto10/4/7,Pedi28/5/5;fixed7/7/10classes,no omissions.",
        "test_gate":"Prepare/inspect validation for all datasets. Commit all scorer/model hashes and fixedbeta before any new test extraction/evaluation. No test-based changes.",
        "accuracy":"All27 main models and18 matched pure-confidence controls; window and frame metrics,all3seeds,pervideo confusion matrices.2000pairedvideo bootstrap.",
        "verification":"Fresh actual-image pipelines for all three main configs and all val/test videos; validate selected indices,pooledfeatures and3seedpredictions.",
        "timing":"Only after accuracy and image verification; shortest/median/longest validation videos,3repeats,5cases (Dense,joint8,joint4,confidence8,confidence4),rotated order.135wholevideo passes. Same warmed filecache,8CPUworkers,FP32,TF32matmuloff/cuDNNon. Both scorer CUDAgraphs andonecurrentGRU resident for every case.",
        "input_reuse":"Chole/Auto share scorer and encoder preprocessed tensor. Pedi staged thumbnail scoring and selected-original encoding. Canonical complete video scoring including0..15trailingframes for original64batchshapes;classify only complete16framewindows.",
        "flops":"Measure CNN/ViT supported-operator costs with identical counter. Separately document CPUjoint math and GRU exclusions; no total-algorithmFLOPs claim.",
        "scope":"Exploratory repeatedly examined tests. Fixedbeta transferred fromPedi gives it prior exposure. No guarantee all six joint results improve.",
    }
    p = RUN/"protocol.json"
    if p.exists():
        assert read(p)==protocol
    else:
        save_json(p,protocol)


def signal_path(d,v):
    return RUN/f"rich/{d}/{v}.npz"


def get_indices(d,v,k,beta):
    ds=ds_for(d)
    if k==16:
        return indices("dense",len(ds.labels(v)),16)
    from vision_joint import select_joint
    with np.load(signal_path(d,v)) as z:
        return select_joint(z["confidence"],z["features"],k,beta)


@torch.inference_mode()
def evaluate(row, split, beta, save=False):
    ds=ds_for(row["dataset"]);model=model_for(row)
    cms=[];fms=[];predictions={}
    stage="dense" if row["k"]==16 else ("joint" if beta else "confidence")
    dest=RUN/f"eval/{ds.name}/{split}/{stage}/{row['config']}/seed{row['seed']}"
    for vid in ds.ids(split):
        ix=get_indices(ds.name,vid,row["k"],beta)
        pooled=ds.feats(vid)[ix].mean(1)
        p=model(torch.from_numpy(pooled).cuda()[None]).argmax(1)[0].cpu().numpy()
        y=ds.labels(vid);wc=confusion(window_labels(y,ds.classes),p,ds.classes)
        fc=confusion(y[:len(p)*16],np.repeat(p,16),ds.classes)
        cms.append(wc);fms.append(fc);predictions[vid]=p
        if save:
            dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/f"{vid}.npz",prediction=p,selected_indices=ix)
    result={**row,"split":split,"stage":stage,"beta":None if row["k"]==16 else beta,
            "window":{m:float(v) for m,v in metric(np.sum(cms,axis=0)).items()},
            "frame":{m:float(v) for m,v in metric(np.sum(fms,axis=0)).items()},
            "source":str(dest/"metrics.json")}
    if save:
        np.savez_compressed(dest/"confusions.npz",window=np.asarray(cms),frame=np.asarray(fms),videos=np.array(ds.ids(split)))
        save_json(dest/"metrics.json",result)
    return result,predictions
