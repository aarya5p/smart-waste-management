# Smart Waste Segregation System Using Deep Learning

Minor project. One student, working with Claude Code. Supervising professor reviews progress.

## Goal
Computer-vision system that identifies waste type, compares several deep-learning models,
is evaluated on real-world images, and runs on a Raspberry Pi that indicates the correct bin.

## Fixed decisions (do not change without asking me)
- Classes (6, from TrashNet): cardboard, glass, metal, paper, plastic, trash
- Classifiers to compare: MobileNetV2, MobileNetV3-Large, EfficientNet-B0, ResNet50, ConvNeXt-Tiny
  (ImageNet-pretrained; frozen feature extraction for all, fine-tuning for 2-3 of them)
- Detection: YOLO nano (Ultralytics, yolo11n or yolov8n) trained on TACO mapped to the 6 classes,
  plus my own annotated mixed-waste photos
- Final system: two-stage (YOLO localizes -> classifier labels each crop), compared against
  end-to-end YOLO
- Deployment target: Raspberry Pi (YOLO exported to NCNN; classifiers to ONNX/TFLite, FP32 and INT8)
- Framework: PyTorch + torchvision/timm, Ultralytics for YOLO
- Methodological reference: Gurcan & Soylu 2026 (docs/reference_paper.pdf)

## Folder layout
- data/raw/trashnet/, data/raw/taco/, data/own_photos/  (never commit; in .gitignore)
- data/processed/   generated datasets (e.g. YOLO format)
- splits/           train/val/test CSVs (filepath,label, paths relative to data dir) - COMMITTED
- src/              all scripts
- runs/             training outputs and checkpoints (never commit)
- results/          metrics, plots, tables used in the report
- docs/             papers, TACO class-mapping table, notes
- LOG.md            experiment log

## Hard rules
1. Never re-split data. Always read splits/*.csv. Existing splits were made by src/prepare_trashnet.py (seed 42).
2. Test sets (TrashNet test split, my real-world test set) are NEVER used for training, tuning,
   early stopping or model selection. Validation set only for those.
3. Photos of the same physical object must all be in the same split.
4. Augmentation only on the training set. Val/test: resize + ImageNet normalization only.
5. Fixed seeds everywhere. Every classifier config is run with seeds 42, 56, 99.
6. After every training or evaluation run, append a row to LOG.md: date, model, settings, seed, key metrics.
7. Every model is scored by the same evaluation script. Metrics: accuracy, macro precision/recall/F1,
   per-class report, confusion matrix; detection: mAP@0.5, mAP@0.5:0.95, per-class AP.
8. Efficiency: params, file size (MB), FLOPs, peak RAM, latency (batch 1, warm-up, median of 100 runs).
   Latency numbers only count if measured on the Raspberry Pi; laptop numbers are labelled as such.
9. Do not delete raw data. Do not install packages globally without asking.
10. Work one phase at a time. Before large changes, show me a plan and wait for approval.
11. Explain what you did and why in plain language after each step - I must be able to explain
    every part of this project in my viva.

## Phases
1. TrashNet cleaning + splits (done: src/prepare_trashnet.py)
2. Classifier train + evaluate scripts; MobileNetV2 baseline
3. All classifiers, 3 seeds, frozen (+ fine-tuned subset); Friedman test
4. TACO: category -> 6-class mapping (docs/taco_mapping.csv, needs professor approval), convert to YOLO
5. YOLO training; add own annotated photos
6. Real-world test set evaluation, broken down by lighting/background/distance/occlusion (metadata CSV)
7. Grad-CAM on misclassified images
8. Export + Raspberry Pi application (camera -> detect -> classify -> LEDs/display) and on-Pi benchmarks
9. Two-stage vs end-to-end comparison; Pareto plot of accuracy vs Pi latency
10. Report figures and tables

## Current status
(update this section as phases finish)
- Phase 1: script written, not yet run on real TrashNet
