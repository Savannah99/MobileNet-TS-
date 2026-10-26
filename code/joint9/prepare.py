"""Complete the missing Chole classification head using train/validation only."""
from concurrent.futures import ThreadPoolExecutor
from base import *
from vision_joint import RichConfidence,extract,select_joint
import torch.nn.functional as F


def head():
    setup();dest=RUN/"cholec_head"
    if (dest/"selection.json").exists():
        assert read(dest/"selection.json")["scorer_sha256"]==sha256(dest/"scorer.pt")
        return
    ds=ds_for("cholec80");dest.mkdir(exist_ok=True)
    m=scorer_for("cholec80",headless=True);rich=RichConfidence(m,cuda_graph=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for split in ("train","val"):
            for vid in ds.ids(split):
                path=dest/f"features/{vid}.npy"
                if not path.exists():
                    h=extract(ds,vid,rich,pool)[:,1:]
                    path.parent.mkdir(parents=True,exist_ok=True);np.save(path,h)
                    print("HEAD_FEATURES",split,vid,len(h),flush=True)
    del rich
    x=torch.from_numpy(np.concatenate([np.load(dest/f"features/{v}.npy") for v in ds.ids("train")])).cuda()
    y=torch.from_numpy(np.concatenate([ds.labels(v) for v in ds.ids("train")])).cuda()
    vx=torch.from_numpy(np.concatenate([np.load(dest/f"features/{v}.npy") for v in ds.ids("val")])).cuda()
    vy=np.concatenate([ds.labels(v) for v in ds.ids("val")])
    assert int(y.min())>=0 and int(y.max())<7 and vy.min()>=0 and vy.max()<7
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    classifier=torch.nn.Linear(64,7).cuda()
    counts=torch.bincount(y,minlength=7).float();freq=counts/counts.sum()
    weights=torch.median(freq[freq>0])/freq.clamp_min(1e-12)
    opt=torch.optim.Adam(classifier.parameters(),lr=1e-3,weight_decay=1e-5)
    def val():
        with torch.inference_mode():
            p=classifier(vx).argmax(1).cpu().numpy()
        return float(metric(confusion(vy,p,7))["jaccard"])
    best=val();best_epoch=0
    best_state={k:v.detach().cpu().clone() for k,v in classifier.state_dict().items()}
    history=[{"epoch":0,"validation_frame_j":best}]
    for epoch in range(1,31):
        order=torch.randperm(len(y),device="cuda");losses=[]
        for batch in order.split(4096):
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(classifier(x[batch]),y[batch],weight=weights)
            loss.backward();opt.step();losses.append(float(loss.detach()))
        score=val()
        if score>best:
            best=score;best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in classifier.state_dict().items()}
        history.append({"epoch":epoch,"validation_frame_j":score,"train_ce":float(np.mean(losses))})
        save_json(dest/"history.json",history);print("HEAD_TRAIN",history[-1],flush=True)
    original=ROOT/"checkpoints/pedi38_extension/mobilenet_ts_plus_cholec80_backbone.pt"
    obj=torch.load(original,map_location="cpu",weights_only=False)
    obj["cls_head"]=best_state
    obj["reconstruction"]="Frozen original backbone,train-only refitted classification head;not original score reproduction."
    torch.save(obj,dest/"scorer.pt")
    save_json(dest/"selection.json",{"state":"complete","epoch":best_epoch,"validation_frame_j":best,
        "committed_utc":now(),"scorer_sha256":sha256(dest/"scorer.pt"),
        "original_sha256":sha256(original),"trained_parameters":455,"epochs":30,
        "feature_files":{str(p):sha256(p) for p in sorted((dest/"features").glob("*.npy"))}})


def rich_split(d,split):
    setup()
    if split=="test":
        assert read(RUN/"selection.json")["state"]=="committed_before_test"
    ds=ds_for(d);scorer=scorer_for(d);rich=RichConfidence(scorer,cuda_graph=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for vid in ds.ids(split):
            dest=signal_path(d,vid);meta=dest.with_suffix(".json")
            if dest.exists() and meta.exists():
                old=read(meta)
                assert old["sha256"]==sha256(dest) and old["scorer_sha256"]==sha256(scorer_path(d))
                continue
            signal=extract(ds,vid,rich,pool)
            assert signal.shape==(len(ds.labels(vid)),65) and np.isfinite(signal).all()
            result={"dataset":d,"split":split,"video":vid,"scorer_sha256":sha256(scorer_path(d))}
            if d!="cholec80":
                old=np.load(ds.scores/vid/"score.npy")
                result["original_score_max_abs"]=float(abs(old-signal[:,0]).max())
                assert result["original_score_max_abs"]<1e-3
                for k in (4,8):
                    assert np.array_equal(indices("tsplus",len(old),k,old),select_joint(signal[:,0],None,k,0))
            dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,confidence=signal[:,0],features=signal[:,1:])
            result["sha256"]=sha256(dest);save_json(meta,result)
            print("RICH",d,split,vid,len(signal),flush=True)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["head","rich"])
    p.add_argument("--dataset",choices=DATASETS);p.add_argument("--split",choices=["val","test"])
    a=p.parse_args()
    if a.stage=="head":head()
    else:rich_split(a.dataset,a.split)
