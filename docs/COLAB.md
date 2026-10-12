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

`dataset-resized.zip` is the file downloaded in Phase 1; its internal layout
(`dataset-resized/cardboard/...` or `cardboard/...`) does not matter, the notebook finds the
class folders either way.

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
4. Edit the `CONFIGS` cell to choose what to run. Default is a single run:
   ```python
   CONFIGS = [{"model": "mobilenet_v2", "mode": "frozen", "seed": 42}]
   ```
   The cell also contains a commented-out full Phase 3 sweep (5 models × 3 seeds, frozen)
   and a fine-tuned subset.
5. **Runtime → Run all.** Watch the training output stream in cell 15.

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

The last cell zips `runs/`, `results/`, `LOG.md` and `pip_freeze_colab.txt` into
`waste_results_<timestamp>.zip` and copies it to `DRIVE_OUTPUT_DIR`.

After downloading it to your laptop:

1. **Copy the new `LOG.md` rows** from the zip into the repo's `LOG.md` and commit them.
   The scripts append a row per run (rule 6), but that happens on the Colab VM - which is
   wiped when the session ends. This is a manual merge step; don't skip it.
2. **Commit `results/`** (metrics, confusion matrices, predictions) - these are the report
   artifacts.
3. **Keep `runs/` locally** if you need the checkpoints for export or Grad-CAM. It stays
   gitignored (checkpoints are too big for git).

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

Colab disconnects idle sessions and caps session length. A single frozen MobileNetV2 run is
quick, but the full 15-run Phase 3 sweep may not fit in one session - run it in chunks by
editing `CONFIGS`, and save a result zip each time (filenames are timestamped, so they don't
overwrite each other).

### Private repo

If the repo is private, set `REPO_IS_PRIVATE = True`. The notebook prompts for a GitHub
personal access token with `getpass` (not printed, not stored in the notebook), uses it only
to build the clone URL, and deletes that URL from memory afterwards.
