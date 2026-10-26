"""Evaluate fixed joint and confidence controls, gated before all test access."""
from base import *


def cases():
    for d in DATASETS:
        for c,k in CONFIGS.items():
            for beta in ([0.] if k==16 else [0.,BETA]):
                for row in model_rows(d,c):
                    yield row,beta


def run_eval(split):
    setup()
    if split=="test":
        s=read(RUN/"selection.json")
        assert s["state"]=="committed_before_test" and s["protocol_sha256"]==sha256(RUN/"protocol.json")
        for d,h in s["scorer_sha256"].items():
            assert sha256(scorer_path(d))==h
    dest=RUN/f"{split}_results.json"
    if dest.exists():
        assert read(dest)["state"]=="complete"
        return
    results=[]
    for row,beta in cases():
        r,_=evaluate(row,split,beta,save=True)
        if split=="test":
            # Unchanged Dense and prior joint Pedi8 must reproduce saved results.
            if row["k"]==16 or (row["dataset"]=="pedi38" and row["k"]==8 and beta==BETA):
                assert abs(r["window"]["jaccard"]-row["prior_window_j"])<1e-14
            if row["dataset"] in ("pedi38","autolaparo") and row["k"]==4 and beta==0:
                assert abs(r["window"]["jaccard"]-row["prior_window_j"])<1e-14
        results.append(r)
        print("EVAL",split,r["dataset"],r["config"],r["stage"],r["seed"],r["window"]["jaccard"],flush=True)
    assert len(results)==45
    save_json(dest,{"state":"complete","records":results,"finished_utc":now(),
                   "protocol_sha256":sha256(RUN/"protocol.json")})
    if split=="val":
        save_json(RUN/"selection.json",{"state":"committed_before_test","committed_utc":now(),
            "beta":BETA,"models_sha256":sha256(RUN/"model_manifest.json"),
            "scorer_sha256":{d:sha256(scorer_path(d)) for d in DATASETS},
            "protocol_sha256":sha256(RUN/"protocol.json"),
            "validation_sha256":sha256(dest),
            "rule":"All six fixed beta0.25 joint variants are evaluated,not chosen by test. Matched beta0 controls also evaluated; a worse joint result will be retained."})


def statistics():
    rows=read(RUN/"test_results.json")["records"];result=[]
    for di,d in enumerate(DATASETS):
        ds=ds_for(d);n=len(ds.ids("test"))
        draw=np.random.default_rng(73103+di).integers(0,n,(2000,n))
        for c in ("ts8","ts4"):
            for control in ("dense","confidence"):
                for level in ("window","frame"):
                    matrices={}
                    for stage,config in (("joint",c),(control,"dense16" if control=="dense" else c)):
                        rr=sorted([r for r in rows if (r["dataset"],r["stage"],r["config"])==(d,stage,config)],
                                  key=lambda r:r["seed"])
                        parts=[]
                        for r in rr:
                            with np.load(Path(r["source"]).parent/"confusions.npz") as z:
                                assert z["videos"].tolist()==ds.ids("test")
                                parts.append(z[level])
                        matrices[stage]=np.stack(parts)
                    diff=np.array([
                        metric(matrices["joint"][s][draw].sum(1))["jaccard"]-
                        metric(matrices[control][s][draw].sum(1))["jaccard"] for s in range(3)]).mean(0)
                    observed=(metric(matrices["joint"].sum(1))["jaccard"]-
                              metric(matrices[control].sum(1))["jaccard"]).mean()
                    result.append({"dataset":d,"config":c,"control":control,"level":level,
                        "difference":float(observed),"ci95":np.quantile(diff,[.025,.975]).tolist(),
                        "iterations":2000,"rng_seed":73103+di})
    save_json(RUN/"bootstrap.json",{"state":"complete","records":result})


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["val","test","statistics"]);a=p.parse_args()
    if a.stage=="statistics":statistics()
    else:run_eval(a.stage)
