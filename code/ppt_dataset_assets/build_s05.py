"""Build matched S05 contact sheets from authentic, consecutive video frames."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/ppt_s05_assets"
PROJECT = ROOT.parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PAGE = "https://commons.wikimedia.org/wiki/File:The_guys_break_dancing_in_the_street_near_the_mall.webm"
LICENSE = "https://creativecommons.org/licenses/by/3.0/"
VIDEO_URL = "https://upload.wikimedia.org/wikipedia/commons/6/6c/The_guys_break_dancing_in_the_street_near_the_mall.webm"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def panel(paths, title, subtitle, source_line, attribution, dest):
    pad, gap, tile_w, tile_h, caption = 24, 14, 456, 257, 42
    header, footer = 124, 100
    width = 2 * pad + 4 * tile_w + 3 * gap
    height = header + 4 * (tile_h + caption) + 3 * gap + footer
    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    fonts = {n: ImageFont.truetype(FONT, n) for n in (26, 29, 32)}
    draw.text((pad, 14), title, font=ImageFont.truetype(BOLD, 46), fill="#1C3077")
    draw.text((pad, 76), subtitle, font=fonts[32], fill="#424242")
    for i, path in enumerate(paths):
        with Image.open(path) as original:
            frame = ImageOps.pad(original.convert("RGB"), (tile_w, tile_h),
                                 method=Image.Resampling.LANCZOS, color="black")
        x = pad + (i % 4) * (tile_w + gap)
        y = header + (i // 4) * (tile_h + caption + gap)
        im.paste(frame, (x, y))
        draw.text((x + 8, y + tile_h + 4), f"{i:02d} s",
                  font=fonts[29], fill="#222222")
    y = height - footer + 8
    draw.text((pad, y), source_line, font=fonts[29], fill="#333333")
    draw.text((pad, y + 43), attribution, font=fonts[26], fill="#555555")
    im.save(dest)
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    left_dir = OUT / "left_frames"
    left_dir.mkdir(exist_ok=True)
    labels_path = PROJECT / "features/vit_small/video02/labels.npy"
    labels = np.load(labels_path)
    indices = np.where(labels == 3)[0]
    run = max(np.split(indices, np.where(np.diff(indices) > 1)[0] + 1), key=len)
    start = int(run[len(run) // 2])
    assert start == 2059 and np.all(labels[start:start + 16] == 3)
    left, records = [], []
    for i in range(16):
        src = PROJECT / f"cholec80/frames_1fps/video02/{start+i:05d}.jpg"
        dst = left_dir / f"{i:02d}.jpg"
        shutil.copy2(src, dst)
        left.append(dst)
        records.append({"relative_second": i, "source_sample_index": start + i,
                        "source_path": str(src), "source_sha256": sha(src),
                        "export_path": str(dst), "phase_label_0based": 3})
    extraction = json.loads((OUT / "right_extraction.json").read_text())
    right = []
    for i, r in enumerate(extraction["records"]):
        assert r["relative_second"] == i
        assert abs(r["decoded_timestamp_ms"] - (i + 18) * 1000) < 1
        path = Path(r["frame_path"])
        right.append(path)
        r["frame_sha256"] = sha(path)
    assert len(right) == 16
    left_image = panel(
        left, "Surgical video", "16 consecutive frames at 1 fps",
        "Cholec80 / video02 | 00:34:19-00:34:34",
        "Gallbladder dissection | Source: Cholec80 / Twinanda et al.",
        OUT / "S05_left_surgical.png")
    right_image = panel(
        right, "Everyday action video: breakdance", "16 consecutive frames at 1 fps",
        "Street breakdance | 00:00:18-00:00:33",
        "KazanCity1 / Wikimedia Commons | CC BY 3.0 | Frames extracted",
        OUT / "S05_right_everyday.png")
    assert left_image.size == right_image.size
    gutter = 48
    combined = Image.new("RGB", (2 * left_image.width + gutter, left_image.height), "white")
    combined.paste(left_image, (0, 0))
    combined.paste(right_image, (left_image.width + gutter, 0))
    combined.save(OUT / "S05_comparison.png")
    combined.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
    combined.save(OUT / "S05_comparison_preview.png")
    manifest = {
        "sampling_fps": 1, "samples_per_side": 16, "panel_size": list(left_image.size),
        "processing": "Real frames; aspect-preserving resizing/letterbox only. No generated frames, color edits, per-frame selection, or time remapping.",
        "left": {"dataset": "Cholec80", "video": "video02", "split": "train",
                 "start_seconds": start, "end_seconds": start + 15,
                 "selection": "Midpoint of longest contiguous gallbladder-dissection label run in training video02; not chosen from prediction scores.",
                 "labels_sha256": sha(labels_path), "records": records},
        "right": {"title": "The guys break dancing in the street near the mall", "author": "KazanCity1",
                  "source_page": PAGE, "source_url": VIDEO_URL,
                  "license": "CC BY 3.0", "license_url": LICENSE,
                  "source_video_sha256": sha(OUT / "source/street_break_dancing.webm"),
                  "start_seconds": 18, "end_seconds": 33,
                  "selection": "A continuous breakdance performance, seconds 18 through 33, selected for visible posture changes and an unobstructed view; fixed 1 fps sampling, not a representative random sample.",
                  "extraction": extraction},
        "interpretation": "Illustrative clips only. Right-side changes include upright, inverted and floor-level body poses during one continuous performance; not a dataset-wide redundancy comparison.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (OUT / "ATTRIBUTION.md").write_text(
        "# S05 source and attribution\n\n"
        "Left: Cholec80, Twinanda et al., EndoNet; University Hospital of Strasbourg / IRCAD. "
        "Local training video02, 00:34:19–00:34:34, gallbladder dissection.\n\n"
        f"Right: *The guys break dancing in the street near the mall*, KazanCity1. Source: `{PAGE}`. "
        f"License: CC BY 3.0, `{LICENSE}`. Original source identified by Commons: "
        "`https://www.youtube.com/watch?v=iVJhL8hSXK8`.\n\n"
        "Changes: extracted 16 frames at 1 fps from 00:00:18–00:00:33; "
        "resized without cropping; added relative timestamps and captions. "
        "Retain this credit and license link with the right panel and any shared derivative. "
        "Attribution does not imply endorsement.\n\n"
        "The right source video is preserved locally for reproducibility. "
        "The segment was chosen for visible body-pose changes and continuous performance, "
        "not as a random sample of everyday videos. These clips illustrate visible differences, "
        "not a quantitative comparison of video domains.\n")
    print(f"Built two {left_image.width} x {left_image.height} panels and comparison previews.")


if __name__ == "__main__":
    main()
