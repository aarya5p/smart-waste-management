# Training on Google Colab

How to run the classifier training on a Colab GPU using
`notebooks/colab_train_classifier.ipynb`.

The notebook never re-splits data and never trains on the test set - it just runs the same
`src/train_classifier.py` and `src/evaluate.py` that run on the laptop, on a faster machine.

---

## 1. One-time setup

### a) Push the code to GitHub

The notebook gets the code by **cloning the repo**, so anything not committed and pushed
simply will not exist on Colab:

```bash
git add src/ splits/ notebooks/ docs/ requirements.txt
git commit -m "Phase 2: classifier training and evaluation scripts"
git push
```

The notebook checks for `src/train_classifier.py`, `src/evaluate.py` and the three
`splits/*.csv` right after cloning, and stops with a clear message if any are missing.

### b) Put the dataset on Drive

Upload the TrashNet images to Drive. **Upload the zip, not the extracted folder.**

Recommended layout:

```
MyDrive/
└── minor_project/
    ├── dataset-resized.zip      <- the 6 class folders
    └── outputs/                 <- result zips land here (created automatically)
```

The zip to upload is **`data/dataset-resized.zip`** in this repo (41 MB, gitignored along with
the rest of `data/`, so it exists only on your machine). Its internal layout
(`dataset-resized/cardboard/...` or `cardboard/...`) does not matter - the notebook finds the
class folders either way.

If it is ever missing, rebuild it from the extracted dataset:

```bash
cd data/raw/trashnet && zip -r ../../dataset-resized.zip cardboard glass metal paper plastic trash
```

Or re-download the original from the TrashNet mirror:

```bash
curl -L -o data/dataset-resized.zip \
  https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip
```

> **Why a zip?** Google Drive is mounted as a network filesystem. Copying 2,527 individual
> small files through it takes minutes; copying one 43 MB zip and unzipping it locally takes
> seconds. Same reason the notebook copies the data to `/content` instead of training
> straight off Drive - reading each image over the network starves the GPU and the epochs
> crawl.

### c) What you do **not** upload

- `runs/` and `results/` - produced on Colab, come back in the result zip
- `data/raw/taco/` - 2.5 GB, only needed for the YOLO phases
- The pretrained ImageNet weights - torchvision downloads them on Colab automatically

---

## 2. Running the notebook

1. Open `notebooks/colab_train_classifier.ipynb` in Colab
   (GitHub → open in Colab, or upload the file).
2. **Runtime → Change runtime type → T4 GPU.** Without this it runs on CPU and takes hours.
3. Edit the **Configuration** cell (cell 0) if your paths differ from the defaults:
   - `REPO_URL` - your GitHub repo
   - `DRIVE_TRASHNET` - the zip (or folder) on Drive
   - `DRIVE_OUTPUT_DIR` - where result zips are written
   - `EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`, `EVAL_SPLITS`
4. **Before Run all, make sure the freezing fix is pushed** (`src/train_classifier.py` with
   `FREEZE_POLICY = "frozen_parts_eval"`, and `src/test_freezing.py`). The clone cell refuses to
   continue without it, and a unit-test cell runs `test_freezing.py` (about 1 min) before any
   training; the sweep only starts if it prints `ALL PASSED`.
5. Two switches in the config cell choose what runs:
   - `RUN_FROZEN = False` (current): the frozen sweep is **not** re-run. The official frozen
     results are the existing runs in the repo (`freeze_policy = bn_adapted`, see CLAUDE.md
     "Limitations for the report"). `True` would re-run all 15 with the fixed code.
   - `RUN_FINETUNE = True` (current): **3 fine-tune runs**, `convnext_tiny`, `mobilenet_v2`,
     `mobilenet_v3_large`, **seed 42 only** (`FINETUNE_SEEDS`), max 50 epochs, patience 7, head lr
     1e-4, unfrozen-backbone lr 1e-5. Add 56 and 99 to `FINETUNE_SEEDS` later to complete rule 5;
     finished runs are skipped.

   Rough time on a T4: about 10-20 min per fine-tune run.
6. **Runtime → Run all.** Watch the training output stream in the run cell.

Each config trains once and is then evaluated on **both splits**, writing to:

```
runs/<run_name>/            best.pt, history.json, config.json
results/<run_name>/val/     metrics.json, confusion_matrix.png, predictions.csv
results/<run_name>/test/    metrics.json, confusion_matrix.png, predictions.csv
```

where `<run_name>` is `<model>_<mode>_seed<seed>`. The session summary table at the end
prints `val_acc`, `val_macro_f1`, `test_acc`, `test_macro_f1` per run.

---

## 3. Getting the results back

**Every finished run is saved to Drive immediately**, under `DRIVE_SWEEP_DIR`
(default `MyDrive/minor_project/sweep/`):

```
sweep/<run_name>/
    runs/<run_name>/       best.pt, history.json, config.json
    results/<run_name>/    val/ and test/ metrics, confusion matrix, predictions
    log_rows.md            the LOG.md rows this run appended (train + 2 eval)
    DONE                   written last; marks the run as complete
sweep/_superseded/<run_name>_<timestamp>/   an older Drive copy that was replaced (see below)
```

The last notebook cell bundles **all** finished runs on Drive into one
`waste_results_<timestamp>.zip` in `DRIVE_OUTPUT_DIR` (with `LOG_rows.md` = every run's rows, and
`pip_freeze_colab.txt`). It is safe to run on a partial sweep, as often as you like.

After downloading the zip to your laptop:

1. **Unzip `runs/` and `results/` into the repo.**
2. **Append `LOG_rows.md` to the repo's `LOG.md`** (same 12-column table; skip exact duplicate
   rows). The Settings column ends in `bn_adapted` (frozen runs, pre-fix code) or
   `frozen_parts_eval` (fixed code), so rows for the same run name from different code versions
   can be told apart. The Colab VM's LOG.md vanishes with the session, so don't skip this (rule 6).
3. **Run the summary** and commit `results/`:
   ```bash
   python src/summarize_results.py
   ```
   It reads every `results/*/val|test/metrics.json` and writes, in `results/`:
   - `summary_frozen` and `summary_finetune` (.csv + .md): mean ± std (sample std over seeds) of
     accuracy and macro F1 per model, val and test as separate tables, ordered by val macro F1
   - `summary_frozen_vs_finetune`: per model, frozen vs fine-tuned side by side, with the change
   - each table shows the freeze policy (`bn_adapted` / `frozen_parts_eval`, read from
     `runs/<run>/config.json`), and the frozen-vs-finetune table adds a caveat when they differ

   Rows with fewer than 3 seeds are marked ⚠ (provisional).
4. **Keep `runs/` locally** for export / Grad-CAM; it stays gitignored.

### Resuming after a disconnect

The VM and its local files are gone, but everything finished is on Drive.

1. Reconnect (set the GPU runtime again if needed) and **re-run the notebook from the top**:
   config → mount Drive → clone/install → copy data → CONFIGS → run cell.
2. The run cell checks each config against Drive. It is skipped (`already on Drive - SKIPPED`)
   only if `sweep/<run_name>/DONE` exists **and** the saved `config.json` has the same max
   epochs, patience, learning rate and `freeze_policy` as requested now. Then the cell continues
   with the first unfinished run.
3. Only the run that was in progress at the moment of the disconnect is lost and restarted from
   scratch (training is not checkpointed mid-run; each run is only a few minutes).
4. A half-copied run can't be mistaken for a finished one: it is copied to `<run_name>.partial`
   and renamed with `DONE` written last. Stale `.partial` folders are overwritten on the retry.
5. A run that *crashes* (e.g. out of memory) doesn't stop the sweep; it is listed at the end as
   failed. Fix the cause and re-run the cell to retry just those.

A Drive folder trained under *different* settings (for example the original BN-adapted frozen
runs, which have no `freeze_policy` and a 30-epoch cap) does not count as done: the run is
redone, and the old folder is **moved** to `sweep/_superseded/<run_name>_<timestamp>/` rather than
deleted. To force a re-run of a finished config, move its folder out of `sweep/`.

---

## 4. Things worth knowing

### Dependencies: why `requirements.txt` isn't installed by default

Colab ships with `torch`/`torchvision` built against its exact CUDA driver, plus numpy,
pandas, matplotlib, scipy and scikit-learn. Forcing our pinned versions would take several
minutes, can replace the CUDA-matched torch with an incompatible build, and usually demands
a runtime restart.

So by default the notebook installs **only what Colab is missing**:
`timm`, `ultralytics`, `imagehash`, `grad-cam`, `fvcore`.

The consequence is honest and worth being able to explain: **library versions on Colab differ
from the laptop's `requirements.txt` pins.** The notebook therefore saves a full
`pip freeze` as `pip_freeze_colab.txt` into the result zip, so any number in the report can
be traced to the exact environment that produced it.

Set `STRICT_REQUIREMENTS = True` in the config cell to force the exact pins instead.

### Validation decides, test only reports (rule 2)

Every run is scored on both splits, so both sets of numbers exist side by side. That makes the
discipline a matter of habit rather than tooling:

- **All decisions come from the validation results only** - hyper-parameters, which models to
  fine-tune, which model to deploy, and any other change to the system.
- **Test results are only reported in the final tables.** They never cause anything to change.
  If a test number makes an earlier choice look wrong, the choice still stands - revising it
  is exactly the leak the rule prevents, and it silently turns the test split into a second
  validation set.

Early stopping and best-checkpoint selection inside `train_classifier.py` always use the
validation set, regardless of what is evaluated afterwards.

A practical consequence for the report: when you quote "the best model", say which split chose
it (val) and which split the headline number comes from (test). Those are two different roles
and examiners do ask.

### Latency numbers (rule 8)

`evaluate.py` measures latency wherever it runs and stamps the result with the device. On
Colab it records `counts_for_report: false`. **Only Raspberry Pi latency belongs in the
report's latency tables** - Colab GPU timings are not a substitute.

### Re-running

Re-running the clone cell pulls the latest commit (`git reset --hard origin/<branch>`), so
you can fix something on the laptop, push, and just re-run that cell. Note this discards any
edits made to the repo *inside* Colab.

The dataset copy cell skips the copy if the class folders are already on local disk, so
re-running it is cheap.

### Session limits

Colab disconnects idle sessions and caps session length. The sweep is built for that: results
are saved per run and finished runs are skipped on restart (see "Resuming after a disconnect"),
so just reconnect and re-run. Keep the browser tab open while it trains.

### Private repo

If the repo is private, set `REPO_IS_PRIVATE = True`. The notebook prompts for a GitHub
personal access token with `getpass` (not printed, not stored in the notebook), uses it only
to build the clone URL, and deletes that URL from memory afterwards.

---

## 5. YOLO detector on TACO (`notebooks/colab_train_yolo.ipynb`)

Trains **yolo11n** (COCO-pretrained) on two TACO datasets, one after the other: `taco_yolo_6class`
(the 6 TrashNet classes) and `taco_yolo_1class` (every box = `waste`). Same images, same split.
Settings: imgsz 640, max 100 epochs, early-stopping patience 20 (on **val**), seed 42, batch 16.
Each finished model is evaluated on **val and test** with `src/evaluate_yolo.py`
(mAP@0.5, mAP@0.5:0.95, per-class AP).

### a) One-time setup

1. On the laptop, build the datasets and the Drive zip (already done once; re-running is safe -
   it reads the committed split instead of making a new one):
   ```bash
   python src/taco_to_yolo.py --zip      # -> data/processed/taco_yolo.zip (~960 MB)
   ```
2. Upload `data/processed/taco_yolo.zip` to `MyDrive/minor_project/taco_yolo.zip`.
   It contains the 6-class dataset complete plus only the *labels* of the 1-class dataset; the
   images are identical, so the notebook recreates them with hard links instead of uploading twice.
3. Commit and push `src/train_yolo.py`, `src/evaluate_yolo.py`, `src/taco_to_yolo.py` and
   `splits/taco_*.txt` - the notebook checks for them after cloning.

### b) Running

Open the notebook, set a **T4 GPU** runtime, **Run all**. The data cell unzips to `/content/datasets/`
and verifies every image against `splits/taco_*.txt` before anything trains.

Where things go on Drive (`MyDrive/minor_project/yolo/`):

```
train/<run_name>/      Ultralytics run folder, written DURING training: weights/last.pt + best.pt,
                       results.csv, plots, args.yaml, train_done.json, log_row_train.md
store/<run_name>/      after evaluation: results/<run_name>/val|test/ + log_rows.md + DONE
```

Run names: `yolo11n_taco6_seed42`, `yolo11n_taco1_seed42`. Rough time on a T4: ~1 h per model
if it runs all 100 epochs (often less with early stopping).

### c) Resuming after a disconnect

Training writes its checkpoint to Drive every epoch, so very little is lost. Reconnect, set the GPU
runtime and **Run all** again:

- a model with `store/<run_name>/DONE` is skipped;
- a model whose `train/<run_name>/weights/last.pt` exists but has no `train_done.json` **resumes
  from its last finished epoch** (Ultralytics `resume=True`, original settings);
- a model with `train_done.json` but no `DONE` (session died during evaluation) skips training and
  just re-runs the evaluation.

To retrain a model from scratch, move its `train/<run_name>/` and `store/<run_name>/` folders away.

### d) Getting the results back

The last cell writes `waste_yolo_results_<timestamp>.zip` to `outputs/` with `runs/yolo/<run>/`,
`results/<run>/val|test/`, `LOG_rows.md` and `pip_freeze_colab_yolo.txt`. Unzip `runs/` and
`results/` into the repo, append `LOG_rows.md` to `LOG.md` (drop exact duplicates only), commit
`results/`.

**Rule 2:** the choice of detector for the two-stage system is made from the **val** mAP only.
**Rule 8:** the speed numbers in `metrics.json` are Colab GPU timings, not Pi latency.

