# MobileNet-TS+: Efficient Surgical Phase Recognition through Confidence-Guided Frame Selection

Code accompanying the ISBME 2026 abstract *"MobileNet-TS+: Efficient Surgical Phase Recognition through Confidence-Guided Frame Selection."*

MobileNet-TS+ scores frames within each temporal window using a lightweight MobileNetV3-Small network and forwards only the top-ranked frames to a frozen ViT-Small encoder, reducing visual-encoding cost while aiming to preserve surgical phase recognition accuracy. The scorer is trained jointly with a phase-classification head and an auxiliary optical-flow-magnitude regression head; only the classification head's confidence is used at inference.

## Scope of this repository

This repository contains the experiment, scoring, and evaluation code behind the results reported in the abstract, across three datasets: Cholec80, AutoLaparo, and PediFundus38.

It does **not** include the scripts that assemble the internal markdown/JSON progress reports used while preparing the submission — those are report-formatting utilities, not part of the method or the evaluation pipeline, and are omitted here to keep the repository focused on reproducible experiment code.

## Contents

```
code/
  common.py                 Shared dataset/model/metric utilities (window labels, GRU model,
                             confusion-matrix macro metrics, JSON/hash helpers) used by the
                             scripts below
  vision.py                 Frame-scorer and ViT encoder loading/inference (confidence scoring,
                             per-video scoring pipeline)
  efficient_inference.py    Batched confidence scoring and image preprocessing for efficient
                             (top-k-only) inference

  joint9/                   Confidence + feature-diversity joint scoring experiment
                             (combined score = confidence + beta * cosine distance to the
                             previously selected frame's MobileNet feature; beta=0.25 for
                             AutoLaparo/PediFundus38)
    base.py                 Shared experiment setup
    prepare.py               Data/checkpoint preparation
    evaluate.py               Evaluation loop
    runner.py                Experiment orchestration
    checks.py                 Consistency checks
    verify_bench.py           Benchmark verification
    vision_joint.py            Joint (confidence + diversity) scoring implementation

  auto_joint_weights/       AutoLaparo top-k=8 fine-tuned-GRU update (mixed-distillation GRU,
                             selected on validation performance)
    run.py                  Experiment runner
    verify_bench.py          Benchmark verification

  k8_alignment/              Cross-dataset top-k=8 alignment experiment
    run.py                  Experiment runner
    summarize.py              Result summarization

  unified_auto/              Shared multi-dataset experiment driver
    experiment.py             Core experiment logic (dataset/scorer/checkpoint resolution)
    run_remaining.py           Orchestration for remaining experiment cells

  ppt_dataset_assets/        Dataset figure assets used in the presentation
    build.py                  Per-dataset overview figure assets
    build_s05.py               Video-clip/frame figure assets
```

## Data

Experiments use three datasets, none of which are redistributed in this repository:

- **Cholec80** -- Twinanda et al., *IEEE Transactions on Medical Imaging*, 2017
- **AutoLaparo** -- Wang et al., MICCAI 2022
- **PediFundus38** -- an in-house pediatric fundus video dataset; not publicly released

Frame features (frozen ViT-Small embeddings), optical-flow magnitudes, and MobileNet-TS+ confidence scores are precomputed and cached to disk by earlier preprocessing steps (not included here); the scripts in `code/` consume those cached `.npy`/`.npz` files. Absolute paths in these scripts (e.g. `/home/.../features/vit_small`) reflect the original compute environment and will need to be adapted to reproduce results elsewhere.

## Known limitation: Cholec80 scorer checkpoint

The original MobileNet-TS+ classification head trained on Cholec80 was not saved as part of the checkpoint (only the backbone, trunk, and auxiliary flow-regression head were preserved). The Cholec80 numbers reported in the abstract come from a cached confidence-score file produced by that original (now-unreproducible) classification head, not from a checkpoint that can be reloaded and rerun end-to-end. A reconstructed classification head trained from the same backbone produces close but not identical scores. This does not affect the AutoLaparo or PediFundus38 results, whose scorers are fully checkpointed.

## Citation

If you use this code, please cite the ISBME 2026 abstract (citation details to be added upon publication).
