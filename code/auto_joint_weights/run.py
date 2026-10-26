"""Validate the previously selected AutoLaparo k8 weights with fixed joint scoring."""
from pathlib import Path
import sys
import shutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/conference/joint9"))
import base as b
import numpy as np
import torch

RUN = ROOT / "outputs/autolaparo_joint_weight_update_20261022"
D = "autolaparo"


def setup():
    b.configure()
    RUN.mkdir(exist_ok=True)
    original = b.model_rows(D, "ts8")
    prior = b.read(ROOT / "outputs/k8_alignment_20261021/selection.json")
    tuned = [{**r, "config": "ts8", "k": 8, "checkpoint_sha256": r["sha256"]}
             for r in prior["datasets"][D]["models"]]
    assert prior["datasets"][D]["recipe"] == "mixed_distill"
    for r in original + tuned:
        assert b.sha256(r["checkpoint"]) == r["checkpoint_sha256"]
    protocol = {
        "dataset": D, "beta": .25, "k": 8, "window": 16, "stride": 16,
        "scorer": str(b.scorer_path(D)), "scorer_sha256": b.sha256(b.scorer_path(D)),
        "candidates": {"original": original, "mixed_distill": tuned},
        "prior_selection_sha256": b.sha256(ROOT / "outputs/k8_alignment_20261021/selection.json"),
        "selection": "Compare entire original and previously validation-selected mixed_distill triplets by mean validation joint J; tie original. Commit before new joint test. No seed selection, beta search, new training or native candidate.",
        "controls": "Unchanged Dense16 and joint k4; same tuned weights with pure confidence; original joint k8.",
        "training_identity": "Existing mixed_distill: 50%TS8/50%bin-random8 training; weightedCE+0.5*T^2*KL, T=2, same-seed Dense teacher on train videos only; selected epochs0/12/19. No teacher at inference.",
        "splits": {"train": b.ds_for(D).ids("train"), "val": b.ds_for(D).ids("val"),
                   "test": b.ds_for(D).ids("test"), "classes": 7},
        "accuracy": "All3seeds, fixed7classes, window/frame metrics, paired video bootstrap2000.",
        "verification": "Selected joint and matched pure actual-image runs on all4val+7test videos; exact indices and all3seed predictions; pooledfeatures/confidence/trunk maxabs<1e-3.",
        "timing": "Five cases Dense16/joint4/originaljoint8/selectedjoint8/selectedpure8, shortest/median/longest validation videos,3rotated rounds,45passes. Same architecture/precision/input pipeline as joint9.",
        "scope": "Exploratory, test sets previously inspected. Extra prior training/validation selection opportunity for k8; improvements cannot be attributed solely to joint selection. Preserve Chole submitted-abstract identity and Pedi results.",
    }
    path = RUN / "protocol.json"
    if path.exists():
        assert b.read(path) == protocol
    else:
        b.save_json(path, protocol)
        for src in (ROOT / "report22.md", ROOT / "outputs/report22/data.json",
                    ROOT / "outputs/report22/latest.json", ROOT / "outputs/report22/audit.json",
                    ROOT / "code/conference/abstract_aligned/build_report.py"):
            shutil.copy2(src, RUN / ("before_" + src.name))
    return protocol


@torch.inference_mode()
def evaluate(row, split, beta, case):
    dest = RUN / f"eval/{split}/{case}/seed{row['seed']}"
    if (dest / "metrics.json").exists():
        saved = b.read(dest / "metrics.json")
        assert saved["checkpoint_sha256"] == row["checkpoint_sha256"] and saved["beta"] == beta
        return saved
    ds = b.ds_for(D)
    model = b.model_for(row)
    cms, fms = [], []
    dest.mkdir(parents=True, exist_ok=True)
    for vid in ds.ids(split):
        ix = b.get_indices(D, vid, row["k"], beta)
        pooled = ds.feats(vid)[ix].mean(1)
        pred = model(torch.from_numpy(pooled).cuda()[None]).argmax(1)[0].cpu().numpy()
        y = ds.labels(vid)
        cms.append(b.confusion(b.window_labels(y, ds.classes), pred, ds.classes))
        fms.append(b.confusion(y[:len(pred)*16], np.repeat(pred, 16), ds.classes))
        np.savez_compressed(dest / f"{vid}.npz", prediction=pred, selected_indices=ix)
    result = {**row, "split": split, "stage": "joint" if beta else "confidence", "beta": beta,
              "case": case, "source": str(dest / "metrics.json"),
              "window": {m: float(v) for m, v in b.metric(np.sum(cms, axis=0)).items()},
              "frame": {m: float(v) for m, v in b.metric(np.sum(fms, axis=0)).items()}}
    np.savez_compressed(dest / "confusions.npz", window=np.array(cms), frame=np.array(fms),
                        videos=np.array(ds.ids(split)))
    b.save_json(dest / "metrics.json", result)
    print("EVAL", split, case, row["seed"], result["window"]["jaccard"], flush=True)
    return result


def main():
    protocol = setup()
    records = []
    for group, beta, case in (("original", .25, "original_joint"),
                              ("mixed_distill", .25, "mixed_joint"),
                              ("mixed_distill", 0., "mixed_confidence")):
        records += [evaluate(r, "val", beta, case) for r in protocol["candidates"][group]]
    means = {c: float(np.mean([r["window"]["jaccard"] for r in records if r["case"] == c]))
             for c in ("original_joint", "mixed_joint", "mixed_confidence")}
    assert abs(means["mixed_confidence"] - .6776249492152555) < 1e-12
    assert abs(means["original_joint"] - .6659626306024925) < 1e-12
    b.save_json(RUN / "val_results.json", {"records": records, "means": means})
    chosen = "mixed_distill" if means["mixed_joint"] > means["original_joint"] else "original"
    selection = {"state": "committed_before_test", "chosen": chosen,
                 "protocol_sha256": b.sha256(RUN / "protocol.json"),
                 "validation_sha256": b.sha256(RUN / "val_results.json"), "means": means,
                 "models": protocol["candidates"][chosen]}
    if (RUN / "selection.json").exists():
        old = b.read(RUN / "selection.json")
        assert all(old[k] == v for k, v in selection.items())
    else:
        b.save_json(RUN / "selection.json", {**selection, "committed_utc": b.now()})
    print("SELECTION", chosen, means, flush=True)
    if chosen == "original":
        b.save_json(RUN / "outcome.json", {"state": "complete", "chosen": chosen,
                    "reason": "Candidate did not improve validation. No new candidate test evaluation.",
                    "validation": means})
        return
    test = []
    for beta, case in ((.25, "mixed_joint"), (0., "mixed_confidence")):
        test += [evaluate(r, "test", beta, case) for r in selection["models"]]
    b.save_json(RUN / "test_results.json", {"records": test,
                "selection_sha256": b.sha256(RUN / "selection.json"), "finished_utc": b.now()})
    print("TEST", {c: float(np.mean([r["window"]["jaccard"] for r in test if r["case"] == c]))
                   for c in ("mixed_joint", "mixed_confidence")}, flush=True)


if __name__ == "__main__":
    main()
