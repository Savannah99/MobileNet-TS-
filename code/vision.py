"""Restore the actual encoder/scorer and their distinct JPEG sources."""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from common import ROOT, DEVICE

TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def encoder_image(ds, vid, i):
    if ds.name == "pedi38":
        return Path("/mnt/ssd4t/mergeResult/PediFundus38/frames_1fps_sata_backup") / vid / f"{i + 1:05d}.jpg"
    return ds.image(vid, i)


def read_image(path, transform=TRANSFORM):
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def read_batch(paths, pool):
    return torch.stack(list(pool.map(read_image, paths)))


class Scorer(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = mobilenet_v3_small(weights=None).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trunk = nn.Sequential(nn.Linear(576, 64), nn.ReLU(inplace=True))
        self.cls_head = nn.Linear(64, classes)
        self.flow_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.trunk(self.pool(self.features(x)).flatten(1))
        return self.cls_head(h), self.flow_head(h).squeeze(-1)


def load_scorer(ds, *, allow_timing_proxy=False, proxy_seed=0):
    """Restore predictions by default; require explicit opt-in for a timing head."""
    name = "mobilenet_ts_plus_pedi38.pt" if ds.name == "pedi38" else "mobilenet_ts_plus_cholec80_backbone.pt"
    path = ROOT / "checkpoints/pedi38_extension" / name
    d = torch.load(path, map_location="cpu", weights_only=False)
    missing_head = "cls_head" not in d
    if missing_head and not allow_timing_proxy:
        raise ValueError(f"Missing saved classification head: {path}. Cached confidence cannot substitute for live scoring.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(proxy_seed)
        m = Scorer(ds.classes)
    for key in ("features", "trunk", "cls_head", "flow_head"):
        if key == "cls_head" and missing_head:
            continue
        getattr(m, key).load_state_dict(d[key])
    m.timing_proxy_head = missing_head
    return m.to(DEVICE).eval(), path


def load_encoder():
    os.environ["HF_HUB_OFFLINE"] = "1"
    import timm
    return timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0).to(DEVICE).eval()


@torch.inference_mode()
def score_video(ds, vid, model, batch=256, workers=8):
    n = len(ds.labels(vid))
    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, n, batch):
            x = read_batch([ds.image(vid, i) for i in range(start, min(n, start + batch))], pool).to(DEVICE)
            out.append(model(x)[0].softmax(1).max(1).values.cpu().numpy())
    return np.concatenate(out)


@torch.inference_mode()
def verify_vision(ds, encoder, scorer):
    checks = []
    for vid in ds.ids("val"):
        n = len(ds.labels(vid))
        ii = [0, n // 2, n - 1]
        x = torch.stack([read_image(encoder_image(ds, vid, i)) for i in ii]).to(DEVICE)
        feat = encoder(x).cpu().numpy()
        e = float(np.abs(feat - ds.feats(vid)[ii]).max())
        s = None
        if scorer is not None:
            # Restore original 64-frame scoring batches, including last-batch
            # shape, to avoid backend algorithm changes masquerading as a
            # missing/mismatched checkpoint.
            conf = []
            with ThreadPoolExecutor(max_workers=8) as pool:
                for i in ii:
                    start = i // 64 * 64
                    paths = [ds.image(vid, j) for j in range(start, min(start + 64, n))]
                    y = read_batch(paths, pool).to(DEVICE)
                    batch_conf = scorer(y)[0].softmax(1).max(1).values.cpu().numpy()
                    conf.append(batch_conf[i - start])
            conf = np.asarray(conf)
            original_conf = np.load(ds.scores / vid / "score.npy")[ii]
            s = float(np.abs(conf - original_conf).max())
        checks.append({"video": vid, "encoder_max_abs": e, "scorer_max_abs": s})
        assert e < 1e-3, (ds.name, vid, "encoder mismatch", e)
        if s is not None:
            assert s < 1e-3, (ds.name, vid, "scorer mismatch", s)
    return checks
