"""Bounded, validation-selected k8 adaptation. Never select on test results."""
from pathlib import Path
import argparse
import dataclasses
import datetime
import json
import random
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "unified_auto"))
from experiment import ds_for, scorer_for, checkpoint_for
from common import (dataset, save_json, sha256, load_gru, GRU, indices,
                    window_labels, confusion, metric, predict)
from vision import score_video
import numpy as np
import torch
import torch.nn.functional as F

RUN = ROOT / "outputs/k8_alignment_20261021"
RECIPES = ("native", "mixed_distill")
NAMES = ("autolaparo", "pedi38")
EPOCHS = 20


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def ds_for_run(name):
    if name == "autolaparo":
        return dataclasses.replace(ds_for("adapt_e07"), scores=RUN / "scores" / name)
    return dataset(name)


def setup():
    RUN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    protocol = {
        "created_utc": now(),
        "purpose": "Exploratory k8 downstream training alignment; no promised test gain.",
        "datasets": list(NAMES), "seeds": [0, 1, 2], "window": 16, "k": 8,
        "fixed": "Existing splits, complete class set, window majority labels, scorer, ViT, pooling, BiGRU architecture and original Dense16 reference.",
        "initialization": "Each candidate starts from the existing k8 BiGRU of the same seed. No use of the k4 weights suggested by prior test crossover.",
        "recipes": {
            "baseline": "Unchanged existing k8 checkpoint; eligible for selection.",
            "native": "Fine-tune using actual TS+ k8 pooled features; train-window median-frequency weighted cross-entropy.",
            "mixed_distill": "For each training window choose TS+8 or bin-random8 with probability 0.5. Same weighted CE plus 0.5 * T^2 * mean-window KL(teacher||student), T=2. Teacher is existing same-seed Dense16, evaluated only on training videos. Validation/inference uses TS+8 exclusively. Teacher absent at inference."
        },
        "training": {"epochs": EPOCHS, "optimizer": "Adam", "lr": 0.0001,
                     "weight_decay": 0.0001, "clip_grad_norm": 1.0,
                     "scheduler": "ReduceLROnPlateau on actual TS+8 validation J, patience5 factor0.5",
                     "loss_weights": "median positive training window class frequency / class frequency; absent classes weight0"},
        "selection": "Include epoch0 in each seed's candidates; select earliest maximal validation-J epoch per seed. For each dataset choose the recipe with highest mean validation J over all three seeds. Ties prefer baseline then native then mixed_distill. Commit both dataset selections with checkpoint hashes before any new test evaluation.",
        "test": "Evaluate only selected recipe per dataset, all three seeds; recompute fixed Dense16 and historical k8 as reproducibility controls. No candidate switch after observing test.",
        "limitations": "Test set has been examined in prior tasks: this is exploratory validation-guided follow-up, not a fresh untouched test claim. TS+ receives extra tuning; original Dense is held fixed so this is not an equal-tuning-budget selector comparison.",
    }
    if not (RUN / "protocol.json").exists():
        save_json(RUN / "protocol.json", protocol)
        shutil.copy2(ROOT / "report.md", RUN / "report_before_tuning.md")
        shutil.copy2(__file__, RUN / "source_at_launch.py")
    else:
        previous = json.loads((RUN / "protocol.json").read_text())
        assert previous["training"] == protocol["training"]
        assert previous["recipes"] == protocol["recipes"]


def prepare_scores():
    ds = ds_for_run("autolaparo")
    original = ds_for("adapt_e07")
    manifest = []
    model = None
    try:
        for split in ("train", "val", "test"):
            for video in ds.ids(split):
                dest = ds.scores / video / "score.npy"
                src = original.scores / video / "score.npy"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    if split != "train":
                        # Existing scores only, no test inference/metric here.
                        dest.symlink_to(src)
                    else:
                        if model is None:
                            model = scorer_for("adapt_e07")
                        np.save(dest, score_video(ds, video, model, batch=64, workers=8))
                        print("TRAIN_SCORES", video, flush=True)
                scores = np.load(dest)
                assert scores.shape == (len(ds.labels(video)),) and np.isfinite(scores).all()
                manifest.append({"video": video, "split": split,
                                 "sha256": sha256(dest), "frames": len(scores)})
        save_json(RUN / "score_manifest.json", {
            "checkpoint": str(checkpoint_for("adapt_e07")),
            "checkpoint_sha256": sha256(checkpoint_for("adapt_e07")),
            "records": manifest})
    finally:
        del model


def load_pool(ds, split, training=False):
    pool = {}
    for video in ds.ids(split):
        f = np.array(ds.feats(video), dtype=np.float32)
        y_frame = ds.labels(video)
        assert len(f) == len(y_frame)
        scores = np.load(ds.scores / video / "score.npy")
        assert len(scores) == len(f) and np.isfinite(scores).all()
        ix = indices("tsplus", len(f), 8, scores)
        y = window_labels(y_frame, ds.classes)
        dense_ix = indices("dense", len(f), 16)
        item = {"x": torch.from_numpy(f[ix].mean(1)).cuda().unsqueeze(0),
                "dense": torch.from_numpy(f[dense_ix].mean(1)).cuda().unsqueeze(0),
                "y": torch.from_numpy(y).cuda(), "indices": ix, "y_frame": y_frame}
        if training:
            item["raw"] = f
        pool[video] = item
    return pool


@torch.no_grad()
def evaluate(model, pool, classes, dense=False):
    model.eval()
    cms, frames, predictions = [], [], {}
    for vid, item in pool.items():
        p = model(item["dense" if dense else "x"]).argmax(1)[0].cpu().numpy()
        y = item["y"].cpu().numpy()
        cms.append(confusion(y, p, classes))
        frames.append(confusion(item["y_frame"][:len(p)*16], np.repeat(p, 16), classes))
        predictions[vid] = p
    cms, frames = np.asarray(cms), np.asarray(frames)
    return {"window": {k: float(v) for k, v in metric(cms.sum(0)).items()},
            "frame": {k: float(v) for k, v in metric(frames.sum(0)).items()}}, cms, frames, predictions


def train_one(ds, train, val, recipe, seed):
    dest = RUN / "training" / ds.name / recipe / f"seed{seed}"
    summary_path = dest / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    dest.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = load_gru(ds, seed, k=8)
    teacher_probs = {}
    if recipe == "mixed_distill":
        teacher = load_gru(ds, seed, dense=True)
        with torch.no_grad():
            for vid, item in train.items():
                teacher_probs[vid] = (teacher(item["dense"]) / 2).softmax(1)
        del teacher
    # Epoch zero is a candidate, preventing forced validation degradation.
    best_j = evaluate(model, val, ds.classes)[0]["window"]["jaccard"]
    initial_j = best_j
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    counts = np.bincount(np.concatenate([v["y"].cpu().numpy() for v in train.values()]),
                         minlength=ds.classes).astype(float)
    weights = np.divide(np.median(counts[counts > 0]), counts,
                        out=np.zeros_like(counts), where=counts > 0)
    weights = torch.tensor(weights, dtype=torch.float32, device="cuda")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=.5)
    history = []
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        losses = []
        for video in random.sample(list(train), len(train)):
            item = train[video]
            x = item["x"]
            if recipe == "mixed_distill":
                ix = indices("random", len(item["raw"]), k=8, rng=rng)
                random_x = torch.from_numpy(item["raw"][ix].mean(1)).cuda().unsqueeze(0)
                use_native = torch.from_numpy(rng.random((1, len(ix), 1)) < .5).cuda()
                x = torch.where(use_native, x, random_x)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits[0].T, item["y"], weight=weights)
            if recipe == "mixed_distill":
                kd = F.kl_div((logits / 2).log_softmax(1), teacher_probs[video],
                              reduction="sum") / logits.shape[-1] * 4
                loss = loss + .5 * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        val_j = evaluate(model, val, ds.classes)[0]["window"]["jaccard"]
        scheduler.step(val_j)
        if val_j > best_j:
            best_j, best_epoch = val_j, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, "val_j": val_j, "best_val_j": best_j,
               "best_epoch": best_epoch, "loss": float(np.mean(losses)),
               "seconds": time.perf_counter() - started, "lr": opt.param_groups[0]["lr"]}
        history.append(row)
        save_json(dest / "history.json", history)
        print("TRAIN", ds.name, recipe, seed, row, flush=True)
    checkpoint = dest / "selected.pt"
    torch.save({"best_state": best_state, "seed": seed, "window": 16, "top_k": 8,
                "recipe": recipe, "epoch": best_epoch, "selection_split": "val"}, checkpoint)
    model.load_state_dict(best_state)
    measured = evaluate(model, val, ds.classes)[0]["window"]["jaccard"]
    assert abs(measured - best_j) < 1e-12
    result = {"dataset": ds.name, "recipe": recipe, "seed": seed,
              "initial_val_j": initial_j, "best_val_j": best_j, "epoch": best_epoch,
              "checkpoint": str(checkpoint), "sha256": sha256(checkpoint),
              "initial_checkpoint_sha256": sha256(ds.checkpoint(seed, k=8))}
    save_json(summary_path, result)
    return result


def train_select():
    if (RUN / "selection.json").exists():
        committed = json.loads((RUN / "selection.json").read_text())
        assert committed["protocol_sha256"] == sha256(RUN / "protocol.json")
        for item in committed["datasets"].values():
            for row in item["models"]:
                assert sha256(row["checkpoint"]) == row["sha256"]
        print("REUSE_COMMITTED_SELECTION", committed["committed_utc"], flush=True)
        return
    selections = {}
    all_rows = []
    for name in NAMES:
        ds = ds_for_run(name)
        train, val = load_pool(ds, "train", True), load_pool(ds, "val")
        means = {}
        candidates = {}
        for recipe in RECIPES:
            rows = [train_one(ds, train, val, recipe, s) for s in range(3)]
            candidates[recipe] = rows
            means[recipe] = float(np.mean([r["best_val_j"] for r in rows]))
            all_rows.extend(rows)
        baseline = [{"checkpoint": str(ds.checkpoint(s, k=8)),
                     "sha256": sha256(ds.checkpoint(s, k=8)), "seed": s,
                     "best_val_j": candidates["native"][s]["initial_val_j"],
                     "epoch": 0} for s in range(3)]
        candidates["baseline"] = baseline
        means["baseline"] = float(np.mean([r["best_val_j"] for r in baseline]))
        order = ["baseline", *RECIPES]
        winner = max(order, key=means.get)
        selections[name] = {"recipe": winner, "mean_val_j": means,
                            "models": candidates[winner]}
        save_json(RUN / f"validation_{name}.json", {
            "means": means, "candidates": candidates, "selected": winner})
        print("VAL_SELECTION", name, winner, means, flush=True)
        del train, val
        torch.cuda.empty_cache()
    save_json(RUN / "selection.json", {
        "state": "committed_before_test", "committed_utc": now(),
        "protocol_sha256": sha256(RUN / "protocol.json"),
        "datasets": selections, "validation_records": all_rows})


def test_selected():
    selection = json.loads((RUN / "selection.json").read_text())
    assert selection["protocol_sha256"] == sha256(RUN / "protocol.json")
    if (RUN / "results.json").exists():
        previous = json.loads((RUN / "results.json").read_text())
        assert previous["selection_sha256"] == sha256(RUN / "selection.json")
        print("REUSE_COMPLETED_TEST", previous["finished_utc"], flush=True)
        return
    rows = []
    for name in NAMES:
        ds = ds_for_run(name)
        test = load_pool(ds, "test")
        chosen = selection["datasets"][name]
        for config in ("dense16", "baseline_k8", "selected_k8"):
            for seed in range(3):
                if config == "selected_k8":
                    path = Path(chosen["models"][seed]["checkpoint"])
                    assert sha256(path) == chosen["models"][seed]["sha256"]
                    model = GRU(ds.classes).cuda().eval()
                    obj = torch.load(path, map_location="cpu", weights_only=False)
                    model.load_state_dict(obj.get("best_state", obj.get("best_single")))
                else:
                    path = ds.checkpoint(seed, dense=config == "dense16", k=8)
                    model = load_gru(ds, seed, dense=config == "dense16", k=8)
                metrics, cms, fms, predictions = evaluate(model, test, ds.classes, dense=config == "dense16")
                if config != "selected_k8":
                    historical_config = "dense16" if config == "dense16" else "ts8"
                    if name == "autolaparo":
                        old_path = ROOT / "outputs/autolaparo_unified_20261021/eval/adapt_e07/test"
                        old_filename = "{video}.npz"
                    else:
                        old_path = ROOT / "outputs/core9_20261021/eval/pedi38"
                        old_filename = "{video}_draw0.npz"
                    old_path = old_path / historical_config / f"seed{seed}"
                    old_metrics = json.loads((old_path / "metrics.json").read_text())
                    for level in ("window", "frame"):
                        for key in ("jaccard", "f1", "accuracy"):
                            assert abs(metrics[level][key] - old_metrics[level][key]) < 1e-12
                    for vid, prediction in predictions.items():
                        with np.load(old_path / old_filename.format(video=vid)) as z:
                            assert np.array_equal(prediction, z["prediction"])
                dest = RUN / "eval" / name / config / f"seed{seed}"
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest / "confusions.npz", window=cms, frame=fms,
                                    videos=np.array(list(test)))
                for vid, prediction in predictions.items():
                    ix = indices("dense", len(test[vid]["y_frame"])) if config == "dense16" else test[vid]["indices"]
                    np.savez_compressed(dest / f"{vid}.npz", prediction=prediction,
                                        y_window=test[vid]["y"].cpu().numpy(), selected_indices=ix)
                row = {"dataset": name, "config": config, "seed": seed,
                       "recipe": chosen["recipe"] if config == "selected_k8" else "unchanged",
                       **metrics, "checkpoint": str(path), "sha256": sha256(path)}
                save_json(dest / "metrics.json", row)
                rows.append(row)
                print("TEST", name, config, seed, metrics, flush=True)
                del model
        del test
    save_json(RUN / "results.json", {"state": "complete", "selection_sha256": sha256(RUN / "selection.json"),
                                   "finished_utc": now(), "records": rows})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "train", "test", "all"])
    args = p.parse_args()
    setup()
    if args.stage in ("prepare", "all"):
        prepare_scores()
    if args.stage in ("train", "all"):
        train_select()
    if args.stage in ("test", "all"):
        test_selected()


if __name__ == "__main__":
    main()
