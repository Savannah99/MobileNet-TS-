"""Actual-image verification and matched timing of the validation-selected weights."""
from concurrent.futures import ThreadPoolExecutor
from run import *
from vision_joint import RichConfidence, process
from efficient_inference import Confidence


def expected(split, case, seed, vid):
    if case in ("mixed_joint", "mixed_confidence"):
        records = b.read(RUN / f"{split}_results.json")["records"]
        r = next(r for r in records if r["case"] == case and r["seed"] == seed)
    else:
        config = {"dense": "dense16", "joint4": "ts4", "original_joint": "ts8"}[case]
        stage = "dense" if case == "dense" else "joint"
        r = next(r for r in b.read(b.RUN / f"{split}_results.json")["records"]
                 if (r["dataset"], r["config"], r["stage"], r["seed"]) == (D, config, stage, seed))
    with np.load(Path(r["source"]).parent / f"{vid}.npz") as z:
        return z["prediction"]


@torch.inference_mode()
def verify():
    setup()
    selected = b.read(RUN / "selection.json")
    assert selected["chosen"] == "mixed_distill"
    dest = RUN / "verify.json"
    result = b.read(dest) if dest.exists() else {"state": "running", "checks": []}
    if result["state"] == "complete":
        return
    ds = b.ds_for(D)
    encoder = b.load_encoder()
    rich = RichConfidence(b.scorer_for(D), cuda_graph=True)
    models = [b.model_for(r) for r in selected["models"]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for split in ("val", "test"):
            for v in ds.ids(split):
                for case, beta in (("mixed_joint", .25), ("mixed_confidence", 0.)):
                    if any((r["split"], r["video"], r["case"]) == (split, v, case)
                           for r in result["checks"]):
                        continue
                    live = process(ds, v, 8, beta, encoder, rich, models[0], pool, collect=True)
                    ix = b.get_indices(D, v, 8, beta)
                    feature = ds.feats(v)[ix].mean(1)
                    row = {"split": split, "video": v, "case": case,
                           "indices_exact": bool(np.array_equal(ix, live["selected_indices"])),
                           "feature_max_abs": float(abs(feature - live["pooled_features"]).max()),
                           "encoded_frames": live["encoded_frames"], "scored_frames": live["scored_frames"],
                           "seeds_exact": []}
                    assert row["indices_exact"] and row["feature_max_abs"] < 1e-3, row
                    assert row["encoded_frames"] == len(ix)*8
                    with np.load(b.signal_path(D, v)) as z:
                        row["confidence_max_abs"] = float(abs(live["signals"][:, 0] - z["confidence"]).max())
                        row["trunk_max_abs"] = float(abs(live["signals"][:, 1:] - z["features"]).max())
                        assert row["confidence_max_abs"] < 1e-3 and row["trunk_max_abs"] < 1e-3, row
                    actual = torch.from_numpy(live["pooled_features"]).cuda()[None]
                    for seed in range(3):
                        pred = live["predictions"] if seed == 0 else models[seed](actual).argmax(1)[0].cpu().numpy()
                        row["seeds_exact"].append(bool(np.array_equal(pred, expected(split, case, seed, v))))
                    assert all(row["seeds_exact"]), row
                    result["checks"].append(row)
                    b.save_json(dest, result)
                    print("VERIFY", split, v, case, flush=True)
                    del actual, live
    assert len(result["checks"]) == 22
    result.update(state="complete", selection_sha256=b.sha256(RUN / "selection.json"),
                  completed_utc=b.now(), inherited_unchanged_dense_k4=str(b.RUN / f"verify/{D}.json"))
    b.save_json(dest, result)


@torch.inference_mode()
def benchmark():
    setup()
    dest = RUN / "benchmark.json"
    if dest.exists():
        assert b.read(dest)["state"] == "complete", "Inspect interrupted timing before retry."
        return
    assert b.read(RUN / "verify.json")["state"] == "complete"
    selected = b.read(RUN / "selection.json")["models"]
    ds = b.ds_for(D)
    encoder = b.load_encoder()
    scorer = b.scorer_for(D)
    rich, plain = RichConfidence(scorer, cuda_graph=True), Confidence(scorer, cuda_graph=True)
    for n in (4, 8, 16, 32, 64, 128):
        x = torch.zeros(n, 3, 224, 224, device="cuda")
        encoder(x)
        encoder(x)
    del x
    cases = [("dense16", "dense", 0., "dense", b.model_rows(D, "dense16")[0]),
             ("ts8", "joint", .25, "mixed_joint", selected[0]),
             ("ts4", "joint", .25, "joint4", b.model_rows(D, "ts4")[0]),
             ("ts8", "confidence", 0., "mixed_confidence", selected[0]),
             ("ts8", "original_joint", .25, "original_joint", b.model_rows(D, "ts8")[0])]
    ordered = sorted(ds.ids("val"), key=lambda v: (len(ds.labels(v)), v))
    videos = [ordered[0], ordered[len(ordered)//2], ordered[-1]]
    old_bench = b.read(b.RUN / f"benchmark/{D}.json")
    result = {"state": "running", "dataset": D, "videos": videos, "records": [],
              "selection_sha256": b.sha256(RUN / "selection.json"),
              "scorer_sha256": b.sha256(b.scorer_path(D)), "flops": old_bench["flops"],
              "flops_provenance": "Reused supported-operator counts from identical scorer/encoder/GRU architectures; only GRU weights changed. Not a fresh count.",
              "flops_source_sha256": b.sha256(b.RUN / f"benchmark/{D}.json"),
              "device": torch.cuda.get_device_name(),
              "memory_scope": old_bench["memory_scope"], "timing_scope": old_bench["timing_scope"]}
    with ThreadPoolExecutor(max_workers=8) as pool:
        files = set()
        for v in videos:
            n = len(ds.labels(v))
            files.update(ds.image(v, i) for i in range(n))
            files.update(b.encoder_image(ds, v, i) for i in range(n//16*16))
        result["warmed_bytes"] = sum(pool.map(lambda p: len(p.read_bytes()), sorted(files)))
        b.save_json(dest, result)
        for repeat in range(3):
            for vi, v in enumerate(videos):
                offset = (repeat*len(videos)+vi) % len(cases)
                for config, stage, beta, case, row in cases[offset:] + cases[:offset]:
                    model = b.model_for(row)
                    model(torch.zeros(1, 8, 384, device="cuda"))
                    k = b.CONFIGS[config]
                    live = process(ds, v, k, beta, encoder, plain if beta == 0 else rich, model, pool)
                    assert np.array_equal(live.pop("selected_indices"), b.get_indices(D, v, k, beta))
                    assert np.array_equal(live.pop("predictions"), expected("val", case, 0, v))
                    live.update(config=config, stage=stage, case=case, repeat=repeat, video=v,
                                predictions_exact=True, indices_exact=True,
                                checkpoint_sha256=row["checkpoint_sha256"])
                    result["records"].append(live)
                    b.save_json(dest, result)
                    print("BENCH", case, v, repeat, live["ms_per_window"], flush=True)
                    del model
    summaries = {}
    for config, stage, beta, case, row in cases:
        records = [r for r in result["records"] if r["case"] == case]
        rounds = [1000*sum(r["seconds"] for r in records if r["repeat"] == i) /
                  sum(r["windows"] for r in records if r["repeat"] == i) for i in range(3)]
        summaries[f"{stage}/{config}"] = {
            "median_ms": float(np.median(rounds)), "repeat_ms": rounds,
            "range_ms": [min(rounds), max(rounds)], "peak_mib": max(r["peak_allocated_mib"] for r in records)}
    assert len(result["records"]) == 45
    result.update(state="complete", summaries=summaries, completed_utc=b.now())
    b.save_json(dest, result)


if __name__ == "__main__":
    verify()
    benchmark()
