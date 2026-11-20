"""Summarise classifier runs: mean/std over seeds of accuracy and macro F1, per model, val and test
shown separately. Reads results/<run_name>/{val,test}/metrics.json (written by evaluate.py).

Usage:  python src/summarize_results.py                 # frozen runs -> results/summary_frozen.csv/.md
        python src/summarize_results.py --mode finetune # -> results/summary_finetune.csv/.md

Std is the SAMPLE standard deviation (ddof=1) over seeds; with a single seed it is shown as n/a.
Models are ordered by mean VAL macro F1 (rule 2: val is what decisions are made from). The test
table uses the same order and is for reporting only.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

SPLITS = ["val", "test"]


def load_runs(results_dir: Path, mode: str) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for split in SPLITS:
            f = run_dir / split / "metrics.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            if m.get("smoke_test") or m.get("mode") != mode:
                continue
            rows.append({"run": m["run_name"], "model": m["model"], "seed": m["seed"], "split": split,
                         "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "split"])
    out = g.agg(n_seeds=("seed", "nunique"), seeds=("seed", lambda s: ",".join(map(str, sorted(s)))),
                acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std")).reset_index()
    order = (out[out.split == "val"].sort_values("f1_mean", ascending=False)["model"].tolist())
    order += [m for m in out["model"].unique() if m not in order]   # models with no val rows (unlikely)
    out["model"] = pd.Categorical(out["model"], order, ordered=True)
    out["split"] = pd.Categorical(out["split"], SPLITS, ordered=True)
    return out.sort_values(["split", "model"]).reset_index(drop=True)


def fmt(mean, std):
    return f"{mean*100:.2f} ± {std*100:.2f}" if pd.notna(std) else f"{mean*100:.2f} (n/a)"


def to_markdown(s: pd.DataFrame, mode: str) -> str:
    lines = [f"# {mode.capitalize()} classifiers: mean ± std over seeds (%)\n",
             "Std = sample std (ddof=1). Models ordered by mean val macro F1. "
             "Val is for decisions; test is for reporting only.\n"]
    for split in SPLITS:
        lines += [f"## {split}\n", "| Model | Seeds | Accuracy | Macro F1 |", "|---|---|---|---|"]
        for r in s[s.split == split].itertuples():
            flag = "" if r.n_seeds >= 3 else " ⚠"
            lines.append(f"| {r.model} | {r.seeds}{flag} | {fmt(r.acc_mean, r.acc_std)} | "
                         f"{fmt(r.f1_mean, r.f1_std)} |")
        lines.append("")
    if (s.n_seeds < 3).any():
        lines.append("⚠ = fewer than 3 seeds finished; means/stds are provisional.")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--mode", default="frozen", choices=["frozen", "finetune"])
    a = p.parse_args()
    rd = Path(a.results_dir)
    df = load_runs(rd, a.mode)
    if df.empty:
        raise SystemExit(f"No {a.mode} metrics found under {rd}/*/(val|test)/metrics.json")
    s = summarise(df)
    csv_path, md_path = rd / f"summary_{a.mode}.csv", rd / f"summary_{a.mode}.md"
    s.to_csv(csv_path, index=False, float_format="%.4f")
    md = to_markdown(s, a.mode)
    md_path.write_text(md)
    print(md)
    print(f"Wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
