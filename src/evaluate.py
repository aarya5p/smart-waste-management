
import argparse
import json
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless: write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader

from train_classifier import (CLASSES, TrashDataset, append_log_row, build_model,
                              build_transforms, pick_device, seed_worker, set_seed)


# ---------------------------------------------------------------- efficiency helpers
def is_raspberry_pi() -> bool:
    """Rule 8: latency only counts when measured on the Pi, so we detect it explicitly."""
    try:
        model = Path("/proc/device-tree/model").read_text(errors="ignore")
        return "raspberry pi" in model.lower()
    except OSError:
        return False


def peak_ram_mb() -> float:
    """Peak resident set size of this whole process. macOS reports bytes, Linux kilobytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 2) if sys.platform == "darwin" else peak / 1024


def count_flops(model, img_size, device):
    """MACs for one forward pass at batch 1, via fvcore. Returns (gmacs, note)."""
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9, "fvcore FlopCountAnalysis: multiply-accumulate ops (MACs), batch 1"
    except Exception as e:                                  # noqa: BLE001
        return None, f"FLOPs unavailable: {e}"


def measure_latency(model, img_size, device, warmup=20, runs=100):
    """Single-image latency: warm up, then take the median of `runs` timed forwards (rule 8)."""
    model.eval()
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    on_pi = is_raspberry_pi()
    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(times.mean()),
        "p95_ms": float(np.percentile(times, 95)),
        "batch_size": 1, "warmup_runs": warmup, "timed_runs": runs,
        "device": str(device), "machine": f"{platform.system()} {platform.machine()}",
        "measured_on_raspberry_pi": on_pi,
        "counts_for_report": on_pi,
        "note": ("Raspberry Pi measurement - valid for the report."
                 if on_pi else
                 "LAPTOP/DEV MACHINE measurement - label as such, NOT a valid Pi latency (rule 8)."),
    }


# ---------------------------------------------------------------- inference
def predict(model, loader, device, limit_batches=0):
    model.eval()
    paths, y_true, y_pred, probs = [], [], [], []
    with torch.no_grad():
        for i, (x, y, p) in enumerate(loader):
            if limit_batches and i >= limit_batches:
                break
            out = model(x.to(device))
            prob = torch.softmax(out, dim=1).cpu().numpy()
            probs.append(prob)
            y_pred.extend(prob.argmax(1).tolist())
            y_true.extend(y.tolist())
            paths.extend(list(p))
    return paths, np.array(y_true), np.array(y_pred), np.concatenate(probs)


def save_confusion_matrix(cm, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint (the one scoring script)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--csv", default="splits/test.csv", help="Split to score")
    p.add_argument("--out_dir", default=None, help="Default: results/<run_name>")
    p.add_argument("--run_name", default=None, help="Default: taken from the checkpoint")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=42, help="Only affects DataLoader determinism")
    p.add_argument("--skip_latency", action="store_true")
    p.add_argument("--log_file", default="LOG.md")
    p.add_argument("--limit_batches", type=int, default=0,
                   help="Debug only: stop after N batches. Marks the run as a smoke test.")
    return p.parse_args()


def main():
    args = parse_args()
    smoke = args.limit_batches > 0
    set_seed(args.seed)
    device = pick_device(args.device)

    # ---- rebuild the exact architecture the checkpoint was trained with
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_name, mode = ckpt["model_name"], ckpt["mode"]
    img_size = ckpt["img_size"]
    if ckpt["classes"] != CLASSES:
        raise SystemExit(f"Checkpoint class order {ckpt['classes']} != {CLASSES}")

    run_name = args.run_name or ckpt.get("run_name") or ckpt_path.parent.name
    if smoke and not run_name.endswith("_smoketest"):
        run_name += "_smoketest"
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # pretrained=False: weights come from the checkpoint, no need to fetch ImageNet ones
    model, _, _ = build_model(model_name, mode, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    print(f"Run: {run_name} | model: {model_name} ({mode}) | trained epoch {ckpt['epoch']}")
    print(f"Scoring {args.csv} on {device}")
    if smoke:
        print(f"SMOKE TEST: only {args.limit_batches} batches, LOG.md will not be touched.")

    # ---- data: eval transform only (resize + ImageNet normalize, rule 4)
    _, eval_tf = build_transforms(img_size)
    ds = TrashDataset(args.csv, args.data_root, eval_tf)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)

    paths, y_true, y_pred, probs = predict(model, loader, device, args.limit_batches)
    print(f"scored {len(y_true)} images")

    # ---- metrics (rule 7: the same set for every model)
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)[2]
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    report = classification_report(y_true, y_pred, labels=present,
                                   target_names=[CLASSES[i] for i in present],
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

    # ---- efficiency (rule 8)
    n_total = sum(p.numel() for p in model.parameters())
    gmacs, flops_note = count_flops(model, img_size, device)
    efficiency = {
        "params_total": n_total,
        "params_trainable_during_training": sum(
            p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint_mb": ckpt_path.stat().st_size / (1024 ** 2),
        "gmacs": gmacs,
        "gflops_2x_macs": None if gmacs is None else 2 * gmacs,
        "flops_note": flops_note,
        "peak_process_ram_mb": peak_ram_mb(),
        "ram_note": "peak RSS of the whole evaluation process on " + str(device),
        "latency": None if args.skip_latency else measure_latency(model, img_size, device),
    }

    metrics = {
        "run_name": run_name, "model": model_name, "mode": mode, "seed": ckpt["seed"],
        "checkpoint": str(ckpt_path), "trained_epoch": ckpt["epoch"],
        "csv": args.csv, "n_images": int(len(y_true)), "device": str(device),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "smoke_test": smoke,
        "accuracy": float(acc),
        "macro_precision": float(macro_p), "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
        "per_class": report,
        "confusion_matrix": {"labels": CLASSES, "matrix": cm.tolist()},
        "efficiency": efficiency,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- predictions.csv (also the input for the Grad-CAM phase)
    pred_df = pd.DataFrame({
        "filepath": paths,
        "true_label": [CLASSES[i] for i in y_true],
        "pred_label": [CLASSES[i] for i in y_pred],
        "correct": y_true == y_pred,
        "confidence": probs.max(1),
    })
    for i, c in enumerate(CLASSES):
        pred_df[f"prob_{c}"] = probs[:, i]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    save_confusion_matrix(cm, out_dir / "confusion_matrix.png",
                          f"{model_name} ({mode}, seed {ckpt['seed']}) - {Path(args.csv).stem}")

    # ---- console summary
    print(f"\naccuracy        {acc:.4f}")
    print(f"macro precision {macro_p:.4f}\nmacro recall    {macro_r:.4f}\nmacro F1        {macro_f1:.4f}")
    print("\nper-class:")
    print(classification_report(y_true, y_pred, labels=present,
                               target_names=[CLASSES[i] for i in present], zero_division=0))
    print(f"params {n_total:,} | checkpoint {efficiency['checkpoint_mb']:.2f} MB | "
          f"GMACs {gmacs if gmacs is None else round(gmacs, 3)} | "
          f"peak RAM {efficiency['peak_process_ram_mb']:.0f} MB")
    if efficiency["latency"]:
        lat = efficiency["latency"]
        print(f"latency (batch 1, median of {lat['timed_runs']}): {lat['median_ms']:.2f} ms "
              f"on {lat['device']} -> {lat['note']}")
    print(f"\nWrote {out_dir}/metrics.json, confusion_matrix.png, predictions.csv")

    # ---- LOG.md (rule 6)
    if smoke:
        print("Smoke test: LOG.md not updated.")
    else:
        lat = efficiency["latency"]
        lat_txt = "latency n/a" if not lat else (
            f"{lat['median_ms']:.1f} ms/img on {lat['device']}"
            f"{'' if lat['counts_for_report'] else ' (laptop, not for report)'}")
        append_log_row(
            Path(args.log_file), datetime.now().strftime("%Y-%m-%d"), model_name,
            f"{mode}, eval on {args.csv}, img{img_size}", ckpt["seed"],
            f"{acc:.4f}", f"{macro_f1:.4f}",
            f"eval | macro P {macro_p:.4f} R {macro_r:.4f} | params {n_total:,} | "
            f"ckpt {efficiency['checkpoint_mb']:.2f} MB | "
            f"GMACs {gmacs if gmacs is None else round(gmacs, 2)} | {lat_txt} | {run_name}",
        )
        print(f"Appended a row to {args.log_file}")


if __name__ == "__main__":
    main()
