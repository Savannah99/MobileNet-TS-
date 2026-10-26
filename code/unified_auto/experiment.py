"""Unify AutoLaparo inputs; select using validation only, retain old measurements."""
import argparse
import dataclasses
import datetime
import json
from pathlib import Path
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "core9"))
import numpy as np
import torch
from common import dataset, save_json, sha256, load_gru, indices, predict, metric, confusion, window_labels
from vision import load_encoder, load_scorer, read_image, encoder_image, Scorer

BASE = ROOT / "outputs/core9_20261021"
RUN = ROOT / "outputs/autolaparo_unified_20261021"
OLD_CHECKPOINT = BASE / "assets/mobilenet_ts_plus_autolaparo.pt"
CONFIGS = [("dense16", "dense", 16), ("ts8", "tsplus", 8), ("ts4", "tsplus", 4)]


def setup():
    RUN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    if not (RUN / "report_before_unified.md").exists():
        shutil.copy2(ROOT / "report.md", RUN / "report_before_unified.md")
    protocol = {
        "change": "Scorer now uses identical aspect-preserving encoder PNG, PIL geometry and normalization; reuse same GPU tensor before selecting ViT inputs.",
        "unchanged": "ViT inputs/features, labels, 10/4/7 video split, 16-frame windows, three existing random-trained BiGRU seeds per budget.",
        "direct_checkpoint": str(OLD_CHECKPOINT), "direct_sha256": sha256(OLD_CHECKPOINT),
        "adaptation_gate": "Only if direct unified validation mean J is below old two-input validation mean J for either k8 or k4. No test evaluation before this decision.",
        "adaptation_recipe": "Initialize old scorer; at most 8 epochs, seed0, Adam lr5e-5 wd1e-5, original train-only phase weights and optical-flow targets, freeze feature blocks before3, keep original loss coefficients. Select among direct and adaptation epochs by mean validation J across k8/k4 and all 3 downstream seeds; ties favor direct/earlier epoch.",
        "test_policy": "Commit selection.json before evaluating the selected unified candidate on test. Do not switch checkpoint based on test.",
        "timing": "Same three validation videos and three rounds; original two-input and new shared-input paths measured together. No training/image export concurrent with timing.",
    }
    if not (RUN / "protocol.json").exists():
        save_json(RUN / "protocol.json", protocol)


def ds_for(candidate):
    ds = dataset("autolaparo")
    if candidate != "baseline":
        ds = dataclasses.replace(ds, frames=encoder_image(ds, ds.ids("val")[0], 0).parent.parent,
                                 scores=RUN / "scores" / candidate)
    return ds


def checkpoint_for(candidate):
    return OLD_CHECKPOINT if candidate in ("baseline", "direct") else RUN / "checkpoints" / f"{candidate}.pt"


def scorer_for(candidate):
    model = Scorer(7)
    obj = torch.load(checkpoint_for(candidate), map_location="cpu", weights_only=False)
    for key in ("features", "trunk", "cls_head", "flow_head"):
        getattr(model, key).load_state_dict(obj[key])
    model.timing_proxy_head = False
    return model.cuda().eval()


@torch.inference_mode()
def score(candidate, split, model=None):
    ds = ds_for(candidate)
    if candidate == "baseline":
        return
    metadata = RUN / f"score_{candidate}_{split}.json"
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        assert previous["checkpoint_sha256"] == sha256(checkpoint_for(candidate)), "Stale scores: checkpoint changed"
        if previous["state"] == "complete":
            assert all(np.load(ds.scores / v / "score.npy").shape == (len(ds.labels(v)),)
                       for v in ds.ids(split))
            return
    owned = model is None
    if owned:
        model = scorer_for(candidate)
    model.eval()
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for vid in ds.ids(split):
            dest = ds.scores / vid / "score.npy"
            n = len(ds.labels(vid))
            if dest.exists():
                assert np.load(dest).shape == (n,)
                continue
            values = []
            for start in range(0, n, 64):
                x = torch.stack(list(pool.map(read_image,
                    [ds.image(vid, i) for i in range(start, min(start + 64, n))]))).cuda()
                values.append(model(x)[0].softmax(1).max(1).values.cpu().numpy())
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.save(dest, np.concatenate(values))
            records.append({"video": vid, "frames": n, "sha256": sha256(dest)})
            print("SCORE", candidate, split, vid, n, flush=True)
    save_json(metadata, {
        "state": "complete", "checkpoint_sha256": sha256(checkpoint_for(candidate)),
        "videos": ds.ids(split), "new_records": records, "batch": 64})
    if owned:
        del model


@torch.inference_mode()
def evaluate(candidate, split):
    ds = ds_for(candidate)
    rows = []
    for config, method, k in CONFIGS:
        for seed in range(3):
            dest = RUN / "eval" / candidate / split / config / f"seed{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "metrics.json").exists():
                rows.append(json.loads((dest / "metrics.json").read_text()))
                continue
            model = load_gru(ds, seed, dense=method == "dense", k=k)
            cms, fms = [], []
            for vid in ds.ids(split):
                labels, features = ds.labels(vid), ds.feats(vid)
                signal = np.load(ds.scores / vid / "score.npy") if method == "tsplus" else None
                ix = indices(method, len(labels), k, signal)
                pred = predict(model, features[ix].mean(1))
                gt = window_labels(labels, 7)
                wc = confusion(gt, pred, 7)
                fc = confusion(labels[:len(pred) * 16], np.repeat(pred, 16), 7)
                cms.append(wc)
                fms.append(fc)
                np.savez_compressed(dest / f"{vid}.npz", prediction=pred, y_window=gt,
                                    selected_indices=ix, window_cm=wc, frame_cm=fc)
            np.savez_compressed(dest / "confusions.npz", window=np.asarray(cms),
                                frame=np.asarray(fms), videos=np.array(ds.ids(split)))
            row = {
                "candidate": candidate, "split": split, "config": config, "seed": seed, "k": k,
                "window": {k: float(v) for k, v in metric(np.sum(cms, axis=0)).items()},
                "frame": {k: float(v) for k, v in metric(np.sum(fms, axis=0)).items()},
                "checkpoint": str(ds.checkpoint(seed, method == "dense", k)),
                "checkpoint_sha256": sha256(ds.checkpoint(seed, method == "dense", k)),
                "scorer_sha256": sha256(checkpoint_for(candidate)) if method == "tsplus" else None,
            }
            save_json(dest / "metrics.json", row)
            rows.append(row)
            print("EVAL", candidate, split, config, seed, row["window"]["jaccard"], flush=True)
            del model
    summary = {c: float(np.mean([r["window"]["jaccard"] for r in rows if r["config"] == c]))
               for c, _, _ in CONFIGS}
    result = {"state": "complete", "records": rows, "mean_window_j": summary,
              "selection_score": float(np.mean([summary["ts8"], summary["ts4"]]))}
    save_json(RUN / f"eval_{candidate}_{split}.json", result)
    return result


def validate():
    setup()
    baseline = evaluate("baseline", "val")
    score("direct", "val")
    direct = evaluate("direct", "val")
    adapt = any(direct["mean_window_j"][c] < baseline["mean_window_j"][c] for c in ("ts8", "ts4"))
    save_json(RUN / "validation_gate.json", {
        "state": "complete", "baseline": baseline["mean_window_j"],
        "direct": direct["mean_window_j"], "adaptation_required": adapt})
    print("GATE", adapt, baseline["mean_window_j"], direct["mean_window_j"], flush=True)
    if not adapt:
        save_json(RUN / "selection.json", {
            "state": "complete", "candidate": "direct", "checkpoint": str(OLD_CHECKPOINT),
            "checkpoint_sha256": sha256(OLD_CHECKPOINT), "validation_score": direct["selection_score"],
            "reason": "Neither validation budget decreased; adaptation gate not triggered.",
            "committed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["validate", "test"])
    args = p.parse_args()
    setup()
    if args.stage == "validate":
        validate()
    else:
        selected = json.loads((RUN / "selection.json").read_text())["candidate"]
        score(selected, "test")
        evaluate(selected, "test")
