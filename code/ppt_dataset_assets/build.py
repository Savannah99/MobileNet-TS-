"""Compose real, consecutive 1 fps frame samples for the dataset slides."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT.parent
OUT = ROOT / "outputs/ppt_dataset_assets"
PEDI = Path("/mnt/ssd4t/mergeResult/PediFundus38")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp(seconds):
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def main():
    jobs = [
        ("cholec80", "Cholec80", "video01",
         PROJECT / "features/vit_small", PROJECT / "cholec80/frames_1fps",
         1, "Calot triangle dissection", 0, "jpg",
         PROJECT / "cholec80/splits/train.txt"),
        ("autolaparo", "AutoLaparo", "AutoLaparo01",
         PROJECT / "features/autolaparo/vit_small",
         Path("/mnt/ssd4t/SurgKeyMAE/core9_autolaparo_encoder_png"),
         1, "Dividing ligament and peritoneum", 0, "png",
         ROOT / "outputs/core9_20261021/assets/splits/train.txt"),
        ("pedi38", "PediFundus38", "video02",
         PEDI / "features/vit_small", PEDI / "frames_1fps_sata_backup",
         5, "Hiatal closure", 1, "jpg", PEDI / "splits/train.txt"),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sampling_fps": 1, "frame_count_per_dataset": 16,
        "selection_rule": (
            "First training video; preselected illustrative phase; longest "
            "contiguous run of its stored label; 16 consecutive seconds "
            "starting at the midpoint of that run. No prediction scores used."
        ),
        "display_processing": (
            "Aspect-preserving resize and letterbox only; no color changes, "
            "interpolation between time points, or generated image content. "
            "Source files are also copied without modification."
        ),
        "time_origin": "Zero-based time within the local sampled video sequence.",
        "datasets": {},
    }
    for key, name, vid, features, source, target, phase, offset, ext, split in jobs:
        assert vid == split.read_text().split()[0], (name, "training identity")
        labels_path = features / vid / "labels.npy"
        labels = np.load(labels_path)
        indices = np.where(labels == target)[0]
        groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        run = max(groups, key=len)
        start = int(run[len(run) // 2])
        assert np.all(labels[start:start + 16] == target)
        dest = OUT / key
        (dest / "frames").mkdir(parents=True, exist_ok=True)
        # This is a contact sheet of actual data, not a synthetic illustration.
        tile_w, tile_h, caption_h, gap, pad, header = 456, 257, 32, 12, 20, 88
        width = 2 * pad + 4 * tile_w + 3 * gap
        height = header + 4 * (tile_h + caption_h) + 3 * gap + pad
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        title_font = ImageFont.truetype(FONT, 28)
        body_font = ImageFont.truetype(FONT, 22)
        draw.text((pad, 10), f"{name} | {vid} | 16 consecutive frames at 1 fps",
                  font=title_font, fill="#1C3077")
        draw.text((pad, 49), f"Phase: {phase}", font=body_font, fill="#333333")
        frame_records = []
        for j in range(16):
            index = start + j
            src = source / vid / f"{index + offset:05d}.{ext}"
            copy = dest / "frames" / src.name
            shutil.copy2(src, copy)
            with Image.open(src) as original:
                size = list(original.size)
                image = ImageOps.pad(original.convert("RGB"), (tile_w, tile_h),
                                     method=Image.Resampling.LANCZOS, color="black")
            x = pad + (j % 4) * (tile_w + gap)
            y = header + (j // 4) * (tile_h + caption_h + gap)
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + tile_h + 3),
                      f"{timestamp(index)}  (+{j:02d} s)",
                      font=body_font, fill="#222222")
            frame_records.append({
                "index_1fps": index, "time_in_sequence": timestamp(index),
                "source_path": str(src), "source_sha256": sha(src),
                "copied_path": str(copy), "source_size": size,
                "label_0based": int(labels[index]),
            })
        path = dest / f"{key}_1fps_16frames.png"
        sheet.save(path)
        manifest["datasets"][key] = {
            "name": name, "video": vid, "split": "train",
            "phase": phase, "labels_path": str(labels_path),
            "labels_sha256": sha(labels_path), "start_seconds": start,
            "end_seconds": start + 15, "contact_sheet": str(path),
            "contact_sheet_sha256": sha(path), "frames": frame_records,
        }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: {"video": v["video"], "start": v["start_seconds"],
                         "end": v["end_seconds"], "image": v["contact_sheet"]}
                      for k, v in manifest["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()
