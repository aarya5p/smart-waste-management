"""Evaluate a trained YOLO detector on ONE split - the single evaluation script for all detectors
(rule 7). Writes results/<run_name>/<split>/metrics.json plus Ultralytics' plots (PR curves,
confusion matrix) and appends one row to LOG.md (rule 6).

Metrics: mAP@0.5, mAP@0.5:0.95, mean precision/recall, and per class: AP@0.5, AP@0.5:0.95,
precision, recall, number of instances. Efficiency: params, weight-file size, GFLOPs.
Latency is NOT measured here: Ultralytics' speed numbers are batch timings on the machine that ran
this (Colab GPU / laptop) and are stored only for reference (rule 8 - only Pi latency counts).

Run every trained detector on BOTH splits (rule 2): val for decisions, test for reporting only.
  python src/evaluate_yolo.py --weights runs/yolo/<run>/weights/best.pt \
      --data data/processed/taco_yolo_6class/data.yaml --split test --run_name <run>
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_classifier import append_log_row  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a YOLO model on one split")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--run_name", required=True)
    p.add_argument("--out_dir", default=None, help="Default: results/<run_name>/<split>")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--log_file", default="LOG.md")
    p.add_argument("--smoke", action="store_true", help="no LOG.md row (for testing the script)")
    return p.parse_args()


def f(x):
    return None if x is None else round(float(x), 6)


def main():
    a = parse_args()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    out_dir = Path(a.out_dir or f"results/{a.run_name}/{a.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    kw = dict(data=a.data, split=a.split, imgsz=a.imgsz, batch=a.batch, plots=True, verbose=True,
              project=str(out_dir.resolve()), name="ultralytics", exist_ok=True)
    if a.device not in (None, ""):
        kw["device"] = a.device
    m = model.val(**kw)

    names = m.names                                   # {id: name}
    present = [int(i) for i in m.ap_class_index]      # classes with >= 1 instance in this split
    nt = getattr(m, "nt_per_class", None)
    per_class = {}
    for cid, cname in names.items():
        if cid in present:
            j = present.index(cid)
            per_class[cname] = {"ap50": f(m.box.ap50[j]), "ap50_95": f(m.box.ap[j]),
                                "precision": f(m.box.p[j]), "recall": f(m.box.r[j]),
                                "instances": None if nt is None else int(nt[cid])}
        else:
            per_class[cname] = {"ap50": None, "ap50_95": None, "precision": None, "recall": None,
                                "instances": 0, "note": "no instances in this split"}

    w = Path(a.weights)
    train_args = (getattr(model, "ckpt", None) or {}).get("train_args", {}) or {}
    try:
        gflops = get_flops(model.model, a.imgsz)
    except Exception:                                  # noqa: BLE001
        gflops = None
    metrics = {
        "run_name": a.run_name, "weights": str(w), "data": a.data, "split": a.split,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"), "imgsz": a.imgsz,
        "trained_from": str(train_args.get("model")), "seed": train_args.get("seed"),
        "smoke_test": a.smoke,
        "map50": f(m.box.map50), "map50_95": f(m.box.map), "precision": f(m.box.mp), "recall": f(m.box.mr),
        "per_class": per_class,
        "efficiency": {
            "params": sum(p.numel() for p in model.model.parameters()),
            "weights_mb": w.stat().st_size / 1024 ** 2,
            "gflops": gflops, "gflops_note": f"Ultralytics get_flops at imgsz {a.imgsz}, batch 1",
            "speed_ms_per_image": m.speed,
            "speed_note": "Ultralytics batch timings on the evaluation machine - NOT a Raspberry Pi "
                          "latency, not for the report (rule 8)",
        },
        "ultralytics_plots": str(Path(m.save_dir)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n{a.run_name} on {a.split}: mAP50 {metrics['map50']:.4f} | mAP50-95 {metrics['map50_95']:.4f}")
    print(f"{'class':12s}{'AP50':>8s}{'AP50-95':>9s}{'inst':>6s}")
    for c, v in per_class.items():
        ap50 = "-" if v["ap50"] is None else f"{v['ap50']:.3f}"
        ap = "-" if v["ap50_95"] is None else f"{v['ap50_95']:.3f}"
        print(f"{c:12s}{ap50:>8s}{ap:>9s}{str(v['instances']):>6s}")
    print(f"Wrote {out_dir / 'metrics.json'} (plots in {m.save_dir})")

    if a.smoke:
        print("Smoke test: LOG.md not updated.")
        return
    dataset = Path(a.data).parent.name
    pc = ", ".join(f"{c} {v['ap50']:.3f}" for c, v in per_class.items() if v["ap50"] is not None)
    append_log_row(
        Path(a.log_file), date=datetime.now().strftime("%Y-%m-%d"), run=a.run_name, type="eval",
        model=Path(str(train_args.get("model", w.stem))).stem,
        mode=f"detect ({dataset})", split=a.split, seed=train_args.get("seed", "-"), accuracy="-", macro_f1="-",
        params=f"{metrics['efficiency']['params']:,}", settings=f"eval on {a.split}, imgsz{a.imgsz}",
        notes=f"mAP50 {metrics['map50']:.4f}, mAP50-95 {metrics['map50_95']:.4f}; P {metrics['precision']:.4f}, "
              f"R {metrics['recall']:.4f}; per-class AP50: {pc}; weights {metrics['efficiency']['weights_mb']:.1f} MB",
    )
    print(f"Appended a row to {a.log_file}")


if __name__ == "__main__":
    main()
