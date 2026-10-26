"""Full image checks and matched timing for all nine primary configurations."""
from concurrent.futures import ThreadPoolExecutor
from base import *
from vision_joint import RichConfidence,process
from efficient_inference import Confidence
from torch.utils.flop_counter import FlopCounterMode


def expected_row(d,split,c,stage,seed):
    return next(r for r in read(RUN/f"{split}_results.json")["records"] if
                (r["dataset"],r["config"],r["stage"],r["seed"])==(d,c,stage,seed))


def cached_prediction(d,split,c,stage,seed,v):
    r=expected_row(d,split,c,stage,seed)
    with np.load(Path(r["source"]).parent/f"{v}.npz") as z:
        return z["prediction"]


@torch.inference_mode()
def verify(d):
    setup();dest=RUN/f"verify/{d}.json"
    if dest.exists() and read(dest)["state"]=="complete":return
    result=read(dest) if dest.exists() else {"state":"running","dataset":d,"checks":[]}
    ds=ds_for(d);encoder=load_encoder();scorer=scorer_for(d)
    rich=RichConfidence(scorer,cuda_graph=True)
    models={(c,r["seed"]):model_for(r) for c in CONFIGS for r in model_rows(d,c)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for split in ("val","test"):
            for v in ds.ids(split):
                for c,k in CONFIGS.items():
                    if any((r["split"],r["video"],r["config"])==(split,v,c) for r in result["checks"]):continue
                    beta=0 if k==16 else BETA;stage="dense" if k==16 else "joint"
                    live=process(ds,v,k,beta,encoder,rich,models[c,0],pool,collect=True)
                    ix=get_indices(d,v,k,beta)
                    feature=ds.feats(v)[ix].mean(1)
                    row={"split":split,"video":v,"config":c,
                         "indices_exact":bool(np.array_equal(ix,live["selected_indices"])),
                         "feature_max_abs":float(abs(feature-live["pooled_features"]).max()),
                         "encoded_frames":live["encoded_frames"],"scored_frames":live["scored_frames"],
                         "seeds_exact":[]}
                    assert row["indices_exact"] and row["feature_max_abs"]<1e-3,row
                    assert row["encoded_frames"]==len(ix)*k
                    if k!=16:
                        with np.load(signal_path(d,v)) as z:
                            signal=live["signals"]
                            row["confidence_max_abs"]=float(abs(signal[:,0]-z["confidence"]).max())
                            row["trunk_max_abs"]=float(abs(signal[:,1:]-z["features"]).max())
                            assert row["confidence_max_abs"]<1e-3 and row["trunk_max_abs"]<1e-3,row
                    actual=torch.from_numpy(live["pooled_features"]).cuda()[None]
                    for seed in range(3):
                        p=live["predictions"] if seed==0 else models[c,seed](actual).argmax(1)[0].cpu().numpy()
                        row["seeds_exact"].append(bool(np.array_equal(p,cached_prediction(d,split,c,stage,seed,v))))
                    assert all(row["seeds_exact"]),row
                    result["checks"].append(row);save_json(dest,result)
                    print("VERIFY",d,split,v,c,flush=True)
                    del actual,live
    assert len(result["checks"])==3*(len(ds.ids("val"))+len(ds.ids("test")))
    result.update(state="complete",selection_sha256=sha256(RUN/"selection.json"),completed_utc=now())
    save_json(dest,result)


@torch.inference_mode()
def benchmark(d):
    setup();dest=RUN/f"benchmark/{d}.json"
    if dest.exists():
        assert read(dest)["state"]=="complete","Preserve/inspect interrupted timing before retrying."
        return
    assert all(read(RUN/f"verify/{x}.json")["state"]=="complete" for x in DATASETS)
    ds=ds_for(d);encoder=load_encoder();scorer=scorer_for(d)
    rich=RichConfidence(scorer,cuda_graph=True);plain=Confidence(scorer,cuda_graph=True)
    with FlopCounterMode(display=False) as fc:
        encoder(torch.zeros(1,3,224,224,device="cuda"))
    vf=fc.get_total_flops()
    with FlopCounterMode(display=False) as fc:
        scorer(torch.zeros(1,3,224,224,device="cuda"))
    sf=fc.get_total_flops()
    # GPU warmup includes the actual encoder batch sizes before any measurements.
    for n in (4,8,16,32,64,128):
        x=torch.zeros(n,3,224,224,device="cuda")
        encoder(x);encoder(x)
    del x
    cases=[("dense16","dense",0.),("ts8","joint",BETA),("ts4","joint",BETA),
           ("ts8","confidence",0.),("ts4","confidence",0.)]
    ordered=sorted(ds.ids("val"),key=lambda v:(len(ds.labels(v)),v))
    videos=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
    result={"state":"running","dataset":d,"videos":videos,"records":[],
            "selection_sha256":sha256(RUN/"selection.json"),"scorer_sha256":sha256(scorer_path(d)),
            "flops":{"encoder_per_frame":vf,"scorer_per_frame":sf,
                     **{c:k*vf+(16*sf if k!=16 else 0) for c,k in CONFIGS.items()}},
            "device":torch.cuda.get_device_name(),"memory_scope":"Encoder,scorer,bothCUDAgraphs andonecurrentGRU resident;Dense includesidle scoring buffers.",
            "timing_scope":"Warmed-file-cache image-to-prediction,original-video extraction/model setup excluded."}
    with ThreadPoolExecutor(max_workers=8) as pool:
        files=set()
        for v in videos:
            n=len(ds.labels(v));files.update(ds.image(v,i) for i in range(n))
            files.update(encoder_image(ds,v,i) for i in range(n//16*16))
        result["warmed_bytes"]=sum(pool.map(lambda p:len(p.read_bytes()),sorted(files)))
        save_json(dest,result)
        for repeat in range(3):
            for vi,v in enumerate(videos):
                offset=(repeat*len(videos)+vi)%len(cases)
                for c,stage,beta in cases[offset:]+cases[:offset]:
                    model=model_for(model_rows(d,c)[0])
                    model(torch.zeros(1,8,384,device="cuda"))
                    live=process(ds,v,CONFIGS[c],beta,encoder,plain if stage=="confidence" else rich,model,pool)
                    assert np.array_equal(live.pop("selected_indices"),get_indices(d,v,CONFIGS[c],beta))
                    assert np.array_equal(live.pop("predictions"),cached_prediction(d,"val",c,stage,0,v))
                    live.update(config=c,stage=stage,repeat=repeat,video=v,predictions_exact=True,indices_exact=True)
                    result["records"].append(live);save_json(dest,result)
                    print("BENCH",d,c,stage,v,repeat,live["ms_per_window"],flush=True)
                    del model
    summaries={}
    for c,stage,beta in cases:
        rows=[r for r in result["records"] if (r["config"],r["stage"])==(c,stage)]
        rounds=[1000*sum(r["seconds"] for r in rows if r["repeat"]==i)/
                sum(r["windows"] for r in rows if r["repeat"]==i) for i in range(3)]
        summaries[f"{stage}/{c}"]={"median_ms":float(np.median(rounds)),"repeat_ms":rounds,
                                  "range_ms":[min(rounds),max(rounds)],
                                  "peak_mib":max(r["peak_allocated_mib"] for r in rows)}
    assert len(result["records"])==45
    result.update(state="complete",summaries=summaries,completed_utc=now());save_json(dest,result)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["verify","benchmark"])
    p.add_argument("dataset",choices=DATASETS);a=p.parse_args()
    if a.stage=="verify":verify(a.dataset)
    else:benchmark(a.dataset)
