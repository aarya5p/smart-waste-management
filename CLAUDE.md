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
   ALL decisions are made from validation results only: hyperparameters, which models to
   fine-tune, which model to deploy, and anything else that changes the system. Test results
   are ONLY reported in the final tables - they never cause anything to change. If a test
   number makes a choice look wrong, the choice still stands; that is the point of the rule.
   Every run is scored on both splits, into results/<run_name>/val/ and results/<run_name>/test/.
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
- Phase 4: DONE. docs/taco_mapping.csv approved. src/taco_to_yolo.py -> data/processed/
  taco_yolo_6class/ (6 TrashNet classes, classifier order) + taco_yolo_1class/ ("waste"), same
  images (hard links). 1498 of 1500 images kept (2 dropped: contain a Battery = EXCLUDE object);
  4776 boxes; 7 boxes clipped to the image. Images re-saved with EXIF rotation applied (152 TACO
  photos are stored sideways; TACO boxes are in the rotated coordinates), max side 1280.
  Split by image 70/15/15 = 1049/225/224 images, near-duplicate photos (pHash <= 5, 3 pairs) kept
  together. SPLIT SEED = 158, FROZEN (splits/taco_{train,val,test}.txt + taco_split_info.json,
  committed; the script reads them unless --resplit). Selection criterion: of seeds 0-999 (959 meet
  the hard constraint "every class >= 10 boxes in val and in test"), pick the one minimising the
  largest |share - 15%| over all 6 classes x {val, test}, share = boxes in split / boxes of that
  class. Best = 2.72 pp (median seed 9.09 pp). Label counts only, chosen before any training.
  It replaced a first seed-42 split (max deviation 12.95 pp, glass val/test 17/71) on 2026-11-20.
  Boxes per class train/val/test (val %, test %):
    cardboard 142/34/30 (16.5, 14.6)   glass 173/45/36 (17.7, 14.2)   metal 393/71/84 (13.0, 15.3)
    paper     99/20/18  (14.6, 13.1)   plastic 1446/350/273 (16.9, 13.2)   trash 1077/231/254 (14.8, 16.3)
    all boxes 3330/751/695. Box-check images: results/taco_check/.
- Phase 5: READY TO RUN. notebooks/colab_train_yolo.ipynb trains yolo11n on taco_yolo_6class then
  taco_yolo_1class (imgsz 640, 100 ep max, patience 20, seed 42, batch 16), resumable via
  last.pt on Drive, then src/evaluate_yolo.py on val + test (mAP50, mAP50-95, per-class AP).
  src/train_yolo.py and src/evaluate_yolo.py smoke-tested on CPU (1 epoch, 5% data). Next: push,
  upload data/processed/taco_yolo.zip to Drive, run on Colab, import. Own annotated photos later.
- Phase 3: frozen sweep DONE (official). 15 runs (5 models x seeds 42/56/99) in runs/, results/,
  LOG.md; results/summary_frozen.csv/.md. These were trained with the pre-fix code and are recorded
  as freeze_policy = bn_adapted ("frozen weights, BatchNorm statistics adapted"): tag added
  retroactively to each runs/*/config.json (with freeze_policy_note) and to the LOG.md Settings
  column. Not re-run with the fixed code, for time. See "Limitations for the report".
  Freezing bug found 2026-11-20 and fixed in src/train_classifier.py (FREEZE_POLICY =
  "frozen_parts_eval": frozen parts kept in eval mode during training; finetune also unfreezes
  mobilenet_v3_large's 960->1280 layer and convnext_tiny's final LayerNorm). Verified by
  src/test_freezing.py (all 5 models, frozen + finetune; a control shows the old behaviour changes
  frozen BN buffers). The fix and the test stay in the code.
  Reproduction run: the frozen sweep was accidentally re-run on Colab with the old code (14 runs,
  waste_results.zip) and gave IDENTICAL results (all 28 val/test metrics + per-epoch losses);
  its 42 LOG.md rows are appended with Notes "reproduction run (old code), identical results".
- Phase 3b: fine-tuning READY TO RUN. Notebook: RUN_FROZEN = False, RUN_FINETUNE = True ->
  3 runs: convnext_tiny, mobilenet_v2, mobilenet_v3_large, SEED 42 ONLY, max 50 epochs,
  patience 7, head lr 1e-4, backbone lr 1e-5, fixed code (frozen_parts_eval). The three were
  picked from frozen VAL results. Next: commit + push (the notebook refuses to run without the
  fix in the cloned code), run on Colab, import, run summarize_results.py (writes
  summary_finetune and summary_frozen_vs_finetune, the latter with a freeze-policy caveat).
- Phase 2: done. src/train_classifier.py and src/evaluate.py written; MobileNetV2 frozen seed 42
  trained on Colab and imported (runs/ + results/mobilenet_v2_frozen_seed42/): val acc 82.4% /
  macro F1 0.801, test acc 81.1% / macro F1 0.784. LOG.md table now has 12 columns
  (Date, Run, Type, Model, Mode, Split, Seed, Accuracy, Macro F1, Params, Settings, Notes), written
  by both scripts. Next: Phase 3 frozen sweep (5 models x seeds 42/56/99).
  notebooks/colab_train_classifier.ipynb + docs/COLAB.md added for GPU training on Colab.
  TACO repo cloned and all 1500 images downloaded (data/raw/taco/, 2.5 GB, 0 corrupt);
  src/taco_categories.py -> docs/taco_categories.csv (60 categories, 4784 annotations).
- Phase 1: done. TrashNet dataset-resized downloaded into data/raw/trashnet/ (2527 images,
  matches expected total). Ran src/prepare_trashnet.py with defaults (seed 42, threshold 5) ->
  splits/{train,val,test}.csv committed. 33 images dropped (28 same-class duplicate groups,
  5 cross-class label-conflict groups); 0 corrupt files. Final counts: train 1745 / val 374 / test 375.

## Limitations for the report
(keep this list current; each item should appear in the report's limitations section)
1. Frozen results are "frozen weights, BatchNorm statistics adapted" (freeze_policy = bn_adapted),
   not pure feature extraction: during training the whole model was in train mode, so BatchNorm
   running mean/var in the frozen backbones of MobileNetV2, MobileNetV3-Large, EfficientNet-B0 and
   ResNet50 adapted to TrashNet, and stochastic depth was active inside frozen blocks (ConvNeXt,
   EfficientNet). This is a known, documented practice, but it is not what "frozen" usually means.
2. ConvNeXt-Tiny uses LayerNorm (no running statistics), so it was truly frozen apart from
   stochastic depth. The 5-model frozen comparison is therefore not perfectly like-for-like:
   part of ConvNeXt's lead could come from this difference rather than the architecture.
3. The freezing bug was found (2026-11-20) and verified with a unit test (src/test_freezing.py,
   including a control proving the test detects the old behaviour). The frozen sweep was NOT
   re-run with the fixed code, for time reasons.
4. Fine-tuning uses the fixed code (frozen_parts_eval). For the BatchNorm models, the frozen vs
   fine-tuned difference therefore mixes the effect of fine-tuning with the BN-statistics
   difference; only for ConvNeXt is it a clean comparison.
5. Fine-tuning was run with seed 42 only (3 runs), a deviation from hard rule 5 (seeds 42/56/99),
   for time. Fine-tuned numbers have no seed std and can't enter a seed-based significance test.
6. Frozen ConvNeXt-Tiny hit the 30-epoch cap (best epoch 30/30, 30/30, 28/30), so its frozen
   scores may be slightly underestimated. Fine-tuning uses a 50-epoch cap.
7. The fine-tune models were chosen from validation results of the bn_adapted frozen sweep.
8. (Positive, for the reproducibility discussion) An accidental re-run of the frozen sweep with
   the same code and seeds reproduced all results exactly (28/28 metrics files, every per-epoch
   loss), so the training pipeline is deterministic.
9. TACO split: chosen by label balance only, before any training (seed 158 of 0-999: smallest
   largest deviation of any class's val/test box share from 15%, = 2.72 pp; constraint >= 10 boxes
   per class in val and test). It replaced an earlier seed-42 split whose glass val/test was 17/71
   boxes. Remaining caveats: the split is by image, so boxes stay clustered in photos - "Broken
   glass" still supplies 32 of the 45 val glass boxes (3 photos) and 18 of the 36 test glass boxes
   (4 photos), and paper has only 20 val / 18 test boxes. Per-class AP for glass and paper is
   therefore noisy. Balancing on the val+test box counts uses no model results, so rule 2 holds.
10. Rule 3 for TACO is enforced with pHash near-duplicate grouping (threshold 5), which catches
   near-identical photos but not different photos of the same litter item from another angle.
