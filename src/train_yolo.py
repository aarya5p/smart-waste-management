"""Train an Ultralytics YOLO detector on one of the TACO YOLO datasets (resumable).

Early stopping and best.pt selection use the VALIDATION split only (Ultralytics validates on the
`val:` entry of data.yaml every epoch); the test split is never touched here (rule 2).

Resumable: outputs go to <project>/<run_name>/. If weights/last.pt exists but the run has not
finished (no train_done.json), training RESUMES from it with the original settings (Ultralytics
resume=True). If train_done.json exists the run is skipped. Point --project at Google Drive on
Colab and a disconnect only loses the epoch in progress.

Appends one row to LOG.md when training finishes (rule 6). --smoke runs a tiny CPU-friendly check
(1 epoch, 5% of the data, imgsz 160) and does not touch LOG.md.

Run:  python src/train_yolo.py --data data/processed/taco_yolo_6class/data.yaml \
          --run_name yolo11n_taco6_seed42 --project runs/yolo
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_classifier import append_log_row  # noqa: E402  same 12-column LOG.md format


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO on a TACO YOLO dataset (resumable)")
    p.add_argument("--data", required=True, help="data.yaml of the dataset")
    p.add_argument("--run_name", required=True, help="e.g. yolo11n_taco6_seed42")
    p.add_argument("--model", default="yolo11n.pt", help="COCO-pretrained starting weights")
    p.add_argument("--project", default="runs/yolo", help="outputs go to <project>/<run_name>/")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20, help="early stopping on val fitness")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default=None, help="'' / None = auto, 0 = first GPU, cpu")
    p.add_argument("--log_file", default="LOG.md")
    p.add_argument("--smoke", action="store_true", help="1 epoch, 5%% data, imgsz 160, no LOG.md row")
    return p.parse_args()


def read_results_csv(path: Path):
    rows = list(csv.DictReader(open(path)))
    return [{k.strip(): v for k, v in r.items()} for r in rows]


def main():
    a = parse_args()
    from ultralytics import YOLO

    run_dir = Path(a.project) / a.run_name
    done = run_dir / "train_done.json"
    last = run_dir / "weights" / "last.pt"
    if done.exists():
        print(f"{a.run_name}: already finished ({done}) - skipping training.")
        return

    t0 = time.time()
    if last.exists():
        print(f"{a.run_name}: resuming from {last} (settings come from the checkpoint)")
        YOLO(str(last)).train(resume=True)
    else:
        overrides = dict(data=a.data, imgsz=a.imgsz, epochs=a.epochs, patience=a.patience, batch=a.batch,
                         seed=a.seed, deterministic=True, workers=a.workers, project=str(Path(a.project).resolve()),
                         name=a.run_name, exist_ok=True, plots=True)
        if a.device not in (None, ""):
            overrides["device"] = a.device
        if a.smoke:
            overrides.update(epochs=1, fraction=0.05, imgsz=160, plots=False)
        print(f"{a.run_name}: training from {a.model} with {overrides}")
        YOLO(a.model).train(**overrides)
    session_s = time.time() - t0

    # ---- summarise from results.csv (one row per epoch, VAL metrics)
    rows = read_results_csv(run_dir / "results.csv")
    k50, k5095 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    best = max(rows, key=lambda r: float(r[k5095]))        # best val mAP50-95 epoch
    total_s = float(rows[-1]["time"]) if "time" in rows[-1] else session_s
    best_pt = run_dir / "weights" / "best.pt"
    model = YOLO(str(best_pt))
    n_params = sum(p.numel() for p in model.model.parameters())
    import yaml
    args_used = yaml.safe_load((run_dir / "args.yaml").read_text())

    summary = {
        "run_name": a.run_name, "finished_at": datetime.now().isoformat(timespec="seconds"),
        "epochs_run": len(rows), "epochs_max": args_used["epochs"], "patience": args_used["patience"],
        "best_val_map50_95_epoch": int(float(best["epoch"])),
        "best_val_map50": float(best[k50]), "best_val_map50_95": float(best[k5095]),
        "train_seconds_total": total_s, "params": n_params,
        "best_pt": str(best_pt), "smoke_test": a.smoke,
        "note": "best.pt is chosen by Ultralytics' val fitness; best_val_* here = epoch with highest val mAP50-95",
        "args": args_used,
    }
    done.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=2, default=str))

    if a.smoke:
        print("Smoke test: LOG.md not updated.")
        return
    dataset = Path(args_used["data"]).parent.name
    settings = (f"{Path(str(args_used['model'])).stem}, data {dataset}, imgsz{args_used['imgsz']}, "
                f"{args_used['epochs']}ep(max), patience{args_used['patience']}, bs{args_used['batch']}, "
                f"optimizer {args_used.get('optimizer', 'auto')}, COCO-pretrained")
    append_log_row(
        Path(a.log_file), date=datetime.now().strftime("%Y-%m-%d"), run=a.run_name, type="train",
        model=Path(str(args_used["model"])).stem, mode=f"detect ({dataset})", split="val",
        seed=args_used["seed"], accuracy="-", macro_f1="-", params=f"{n_params:,}", settings=settings,
        notes=f"best val mAP50-95 epoch {summary['best_val_map50_95_epoch']}/{len(rows)}; "
              f"val mAP50 {summary['best_val_map50']:.4f}, mAP50-95 {summary['best_val_map50_95']:.4f}; "
              f"{total_s / 60:.1f} min",
    )
    # Keep a copy of the row next to the weights (on Drive in Colab), so it survives even if the
    # session dies before the notebook has saved this run's LOG rows.
    row = Path(a.log_file).read_text().rstrip("\n").split("\n")[-1]
    (run_dir / "log_row_train.md").write_text(row + "\n")
    print(f"Appended a row to {a.log_file} (copy in {run_dir / 'log_row_train.md'})")


if __name__ == "__main__":
    main()
