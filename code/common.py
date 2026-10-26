"""Shared, import-safe conference experiment definitions."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".runtime/conference"))
PROJECT = ROOT.parent
torch.set_num_threads(4)
WINDOW = 16
DEVICE = "cuda"


@dataclass
class Dataset:
    name: str
    classes: int
    features: Path
    frames: Path
    splits: Path
    scores: Path
    flow: Path
    first_frame: int
    csv: Path

    def ids(self, split):
        return (self.splits / f"{split}.txt").read_text().split()

    def labels(self, vid):
        return np.load(self.features / vid / "labels.npy").astype(np.int64)

    def feats(self, vid):
        return np.load(self.features / vid / "features.npy", mmap_mode="r")

    def image(self, vid, index):
        return self.frames / vid / f"{index + self.first_frame:05d}.jpg"

    def checkpoint(self, seed, dense=False, k=8):
        if self.name == "pedi38":
            name = (f"ckpt_uniform_gru_pedi38_w16_seed{seed}.pt" if dense
                    else f"ckpt_bigru_pedi38_w16_topk{k}_seed{seed}.pt")
            return ROOT / "checkpoints/pedi38_extension" / name
        if dense:
            # Same training recipe as the historical uniform baseline, with
            # persistent weights. E0 verifies predictions against original logs.
            return ROOT / "checkpoints/all_frame_ablation" / f"ckpt_bigru_allframe_cholec80_seed{seed}.pt"
        return ROOT / "checkpoints/main_experiment" / f"ckpt_bigru_cholec80_w16_topk{k}_seed{seed}.pt"


def dataset(name):
    if name == "cholec80":
        return Dataset(name, 7, PROJECT / "features/vit_small",
                       PROJECT / "cholec80/frames_1fps", PROJECT / "cholec80/splits",
                       PROJECT / "features/mobilenet_multitask_score",
                       PROJECT / "features/flow_mag", 0,
                       PROJECT / "cholec80/cholec80_1fps.csv")
    if name == "pedi38":
        base = Path("/mnt/ssd4t/mergeResult/PediFundus38")
        return Dataset(name, 10, base / "features/vit_small",
                       PROJECT / "pedi38_frames_1fps", base / "splits",
                       base / "features/mobilenet_multitask_ft_score",
                       base / "features/flow_mag", 1,
                       base / "pedifundus38_1fps.csv.bak_wrong_eye_labels")
    raise ValueError(name)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class GRU(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.proj = nn.Linear(384, 256)
        self.gru = nn.GRU(256, 256, 2, batch_first=True, bidirectional=True, dropout=.3)
        self.head = nn.Linear(512, classes)

    def forward(self, x):
        h, _ = self.gru(self.proj(x))
        return self.head(h).permute(0, 2, 1)


def load_gru(ds, seed, dense=False, k=8):
    checkpoint = ds.checkpoint(seed, dense, k)
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = GRU(ds.classes)
    model.load_state_dict(obj.get("best_state", obj.get("best_single")))
    return model.to(DEVICE).eval()


def window_labels(labels, classes):
    n = len(labels) // WINDOW
    return np.array([np.bincount(x, minlength=classes).argmax()
                     for x in labels[:n * WINDOW].reshape(n, WINDOW)], dtype=np.int64)


def indices(method, length, k=8, signal=None, rng=None):
    n = length // WINDOW
    starts = np.arange(n)[:, None] * WINDOW
    if method == "dense":
        return starts + np.arange(WINDOW)[None, :]
    lo = np.arange(k) * WINDOW // k
    hi = (np.arange(k) + 1) * WINDOW // k
    if method == "fixed":
        return starts + lo + (hi - lo) // 2
    if method == "fixed0":
        return starts + lo
    if method == "random":
        return starts + rng.integers(lo, hi, size=(n, k))
    if method == "tsplus":
        local = np.asarray(signal[:n * WINDOW]).reshape(n, WINDOW)
        return starts + np.stack([a + local[:, a:b].argmax(1) for a, b in zip(lo, hi)], 1)
    if method not in ("mg_flow", "mg_ssim", "mg_ssim_unique", "mg_ssim_source"):
        raise ValueError(method)
    out = []
    for s in range(0, n * WINDOW, WINDOW):
        motion = np.clip(np.asarray(signal[s:s + WINDOW], dtype=np.float64), 0, None)
        if method == "mg_flow":
            motion += 1e-6
            cdf = np.cumsum(motion) / motion.sum()
            chosen = np.searchsorted(cdf, (np.arange(k) + .5) / k).clip(0, WINDOW - 1)
        else:
            # Official MG image branch: sqrt motion + nearest CDF midpoint.
            # Reset first-frame motion at a window boundary: 15 intra-window
            # transitions, each aligned to its destination frame.
            motion[0] = 0
            motion = np.sqrt(motion)
            if motion.sum() <= 1e-12:
                out.append(s + lo + (hi - lo) // 2)
                continue
            cdf = np.cumsum(motion[1:]) / motion.sum()
            chosen = np.abs(cdf[:, None] - (np.arange(k) + .5)[None, :] / k).argmin(0)
            # Official one-based output j+1 names JPEG frame j (zero-based).
            # Keep the destination-aligned v1 as an explicit sensitivity check.
            if method != "mg_ssim_source":
                chosen = chosen + 1
        if method in ("mg_flow", "mg_ssim_unique"):
            chosen = sorted(set(chosen.tolist()))
            chosen += [i for i in range(WINDOW) if i not in chosen][:k - len(chosen)]
        out.append(s + np.asarray(chosen))
    return np.stack(out)


def confusion(y, pred, classes):
    return np.bincount(classes * np.asarray(y) + np.asarray(pred),
                       minlength=classes * classes).reshape(classes, classes)


def metric(cm):
    """Accepts (..., C, C), always uses the fixed class set."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    gt, pd = cm.sum(-1), cm.sum(-2)
    union = gt + pd - tp
    j = np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union != 0)
    den = gt + pd
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0)
    return {"jaccard": j.mean(-1), "f1": f1.mean(-1),
            "accuracy": tp.sum(-1) / cm.sum(axis=(-2, -1)).clip(min=1)}


@torch.inference_mode()
def predict(model, pooled):
    x = torch.from_numpy(np.asarray(pooled, dtype=np.float32)).unsqueeze(0).to(DEVICE)
    return model(x).argmax(1)[0].cpu().numpy()


def eval_signals(ds, models, video_ids, signals, k=8):
    vals = []
    for model in models:
        cms = []
        for vid in video_ids:
            f, y = ds.feats(vid), ds.labels(vid)
            idx = indices("tsplus", len(f), k, signals[vid])
            p = predict(model, f[idx].mean(1))
            cms.append(confusion(window_labels(y, ds.classes), p, ds.classes))
        vals.append(float(metric(np.sum(cms, axis=0))["jaccard"]))
    return vals
