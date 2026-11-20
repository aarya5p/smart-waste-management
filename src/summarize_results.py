"""Summarise classifier runs: mean/std over seeds of accuracy and macro F1, per model, val and test
shown separately. Reads results/<run_name>/{val,test}/metrics.json (written by evaluate.py).

Usage:  python src/summarize_results.py            # every mode that has results, plus the comparison
        python src/summarize_results.py --mode frozen   # only results/summary_frozen.csv/.md

Outputs (in results/):
  summary_frozen.csv/.md, summary_finetune.csv/.md   mean/std over seeds per model, val and test separate
  summary_frozen_vs_finetune.csv/.md                 per model: frozen vs fine-tuned macro F1 and
                                                     accuracy side by side, val and test separate
                                                     (only models that have BOTH modes)
  summary_bn_ablation.csv/.md                        per model: "BN adapted" (archived old frozen runs,
                                                     results_archive/frozen_bn_adapted/) vs "truly
                                                     frozen" (current frozen runs), val and test separate

Std is the SAMPLE standard deviation (ddof=1) over seeds; with a single seed it is shown as n/a.
Models are ordered by mean VAL macro F1 (rule 2: val is what decisions are made from). The test
table uses the same order and is for reporting only.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

SPLITS = ["val", "test"]


def freeze_policy_of(runs_dir: Path, run_name: str) -> str:
    """freeze_policy from runs/<run>/config.json: 'frozen_parts_eval' (fixed code) or 'bn_adapted'
    (pre-fix runs, tagged retroactively). 'unknown' if the checkpoint folder isn't on this machine."""
    try:
        return json.loads((runs_dir / run_name / "config.json").read_text()).get("freeze_policy", "unknown")
    except (OSError, ValueError):
        return "unknown"


def load_runs(results_dir: Path, mode: str, runs_dir: Path = Path("runs")) -> pd.DataFrame:
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
                         "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                         "freeze_policy": freeze_policy_of(runs_dir, m["run_name"])})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "split"])
    out = g.agg(n_seeds=("seed", "nunique"), seeds=("seed", lambda s: ",".join(map(str, sorted(s)))),
                acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                freeze_policy=("freeze_policy", lambda s: "+".join(sorted(set(s))))).reset_index()
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
             "Val is for decisions; test is for reporting only.\n",
             "Freeze policy: `bn_adapted` = pre-fix code, backbone weights frozen but BatchNorm running "
             "statistics adapted to TrashNet (ConvNeXt has LayerNorm, so it was truly frozen apart from "
             "stochastic depth); `frozen_parts_eval` = fixed code, frozen parts kept in eval mode.\n"]
    for split in SPLITS:
        lines += [f"## {split}\n", "| Model | Seeds | Freeze policy | Accuracy | Macro F1 |",
                  "|---|---|---|---|---|"]
        for r in s[s.split == split].itertuples():
            flag = "" if r.n_seeds >= 3 else " ⚠"
            lines.append(f"| {r.model} | {r.seeds}{flag} | {r.freeze_policy} | "
                         f"{fmt(r.acc_mean, r.acc_std)} | {fmt(r.f1_mean, r.f1_std)} |")
        lines.append("")
    if (s.n_seeds < 3).any():
        lines.append("⚠ = fewer than 3 seeds finished; means/stds are provisional.")
    return "\n".join(lines) + "\n"


def compare(left: pd.DataFrame, right: pd.DataFrame, lk: str, rk: str) -> pd.DataFrame:
    """One row per (split, model) present in both summaries. delta = right - left (means).
    Column suffixes are _<lk> and _<rk>. Ordered by the RIGHT side's val macro F1."""
    cols = ["model", "split", "n_seeds", "freeze_policy", "acc_mean", "acc_std", "f1_mean", "f1_std"]
    lft = left[cols].astype({"model": str, "split": str})
    rgt = right[cols].astype({"model": str, "split": str})
    m = lft.merge(rgt, on=["model", "split"], suffixes=(f"_{lk}", f"_{rk}"))
    m["acc_delta"] = m[f"acc_mean_{rk}"] - m[f"acc_mean_{lk}"]
    m["f1_delta"] = m[f"f1_mean_{rk}"] - m[f"f1_mean_{lk}"]
    order = m[m.split == "val"].sort_values(f"f1_mean_{rk}", ascending=False)["model"].tolist()
    order += [x for x in m["model"].unique() if x not in order]
    m["model"] = pd.Categorical(m["model"], order, ordered=True)
    m["split"] = pd.Categorical(m["split"], SPLITS, ordered=True)
    return m.sort_values(["split", "model"]).reset_index(drop=True)


def compare_markdown(m: pd.DataFrame, lk: str, rk: str, lname: str, rname: str,
                     title: str, extra_note: str = "") -> str:
    lines = [f"# {title}: mean ± std over seeds (%)\n",
             f"Δ = {rname} mean - {lname} mean (percentage points). Std = sample std (ddof=1). "
             f"Models ordered by {rname} val macro F1. Val is for decisions; test is for reporting only.\n"]
    if extra_note:
        lines.append(extra_note + "\n")
    for split in SPLITS:
        lines += [f"## {split}\n",
                  f"| Model | {lname} acc | {rname} acc | Δ acc | {lname} macro F1 | {rname} macro F1 | Δ F1 |",
                  "|---|---|---|---|---|---|---|"]
        for r in m[m.split == split].to_dict("records"):
            flag = "" if min(r[f"n_seeds_{lk}"], r[f"n_seeds_{rk}"]) >= 3 else " ⚠"
            lines.append(f"| {r['model']}{flag} | {fmt(r[f'acc_mean_{lk}'], r[f'acc_std_{lk}'])} | "
                         f"{fmt(r[f'acc_mean_{rk}'], r[f'acc_std_{rk}'])} | {r['acc_delta']*100:+.2f} | "
                         f"{fmt(r[f'f1_mean_{lk}'], r[f'f1_std_{lk}'])} | "
                         f"{fmt(r[f'f1_mean_{rk}'], r[f'f1_std_{rk}'])} | {r['f1_delta']*100:+.2f} |")
        lines.append("")
    if ((m[f"n_seeds_{lk}"] < 3) | (m[f"n_seeds_{rk}"] < 3)).any():
        lines.append("⚠ = fewer than 3 seeds finished on at least one side; provisional.")
    return "\n".join(lines) + "\n"


def write_comparison(rd: Path, stem: str, m: pd.DataFrame, md: str):
    csv_path, md_path = rd / f"{stem}.csv", rd / f"{stem}.md"
    m.to_csv(csv_path, index=False, float_format="%.4f")
    md_path.write_text(md)
    print(md)
    print(f"Wrote {csv_path} and {md_path}\n")


BN_ABLATION_NOTE = (
    "**BN adapted** = archived original frozen sweep: backbone weights frozen, but the whole model "
    "was in train mode, so BatchNorm running stats in the frozen backbone adapted to TrashNet and "
    "stochastic depth was active; max 30 epochs. **Truly frozen** = frozen parts kept in eval mode "
    "(BN stats fixed at ImageNet values); max 50 epochs. Caveat: convnext_tiny has no BatchNorm "
    "(LayerNorm), so its difference comes only from stochastic depth being off and from the higher "
    "epoch cap (it hit the 30-epoch cap in the BN-adapted runs) - it is not a BN effect.")


def write_mode(rd: Path, mode: str):
    df = load_runs(rd, mode)
    if df.empty:
        print(f"(no {mode} results yet - skipped)")
        return None
    s = summarise(df)
    csv_path, md_path = rd / f"summary_{mode}.csv", rd / f"summary_{mode}.md"
    s.to_csv(csv_path, index=False, float_format="%.4f")
    md = to_markdown(s, mode)
    md_path.write_text(md)
    print(md)
    print(f"Wrote {csv_path} and {md_path}\n")
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--mode", default="all", choices=["all", "frozen", "finetune"])
    p.add_argument("--archive_dir", default="results_archive/frozen_bn_adapted",
                   help="Archived BN-adapted frozen runs, for the BN ablation table")
    a = p.parse_args()
    rd = Path(a.results_dir)
    modes = ["frozen", "finetune"] if a.mode == "all" else [a.mode]
    out = {mode: write_mode(rd, mode) for mode in modes}
    if all(v is None for v in out.values()):
        raise SystemExit(f"No metrics found under {rd}/*/(val|test)/metrics.json")
    if a.mode == "all" and out["frozen"] is not None and out["finetune"] is not None:
        m = compare(out["frozen"], out["finetune"], "frozen", "finetune")
        if m.empty:
            print("(no model has both frozen and finetune results yet - comparison skipped)")
        else:
            mixed = (m["freeze_policy_frozen"] != m["freeze_policy_finetune"]).any()
            note = ("**Caveat:** frozen and fine-tuned runs used different freeze policies "
                    "(frozen: `" + "/".join(sorted(set(m["freeze_policy_frozen"]))) + "`, fine-tuned: `"
                    + "/".join(sorted(set(m["freeze_policy_finetune"]))) + "`). For BatchNorm models the "
                    "Δ therefore mixes the effect of fine-tuning with the BN-statistics difference; "
                    "for convnext_tiny (LayerNorm) it does not.") if mixed else ""
            write_comparison(rd, "summary_frozen_vs_finetune", m, compare_markdown(
                m, "frozen", "finetune", "Frozen", "Fine-tuned", "Frozen vs fine-tuned", note))

    # BN ablation: archived "BN adapted" frozen runs vs current truly-frozen runs
    arch = Path(a.archive_dir)
    if a.mode in ("all", "frozen") and out.get("frozen") is not None and arch.is_dir():
        old = load_runs(arch, "frozen")
        if old.empty:
            print(f"(no archived frozen runs in {arch} - BN ablation skipped)")
        else:
            m = compare(summarise(old), out["frozen"], "bn_adapted", "truly_frozen")
            write_comparison(rd, "summary_bn_ablation", m, compare_markdown(
                m, "bn_adapted", "truly_frozen", "BN adapted", "Truly frozen",
                "Ablation: BN adapted vs truly frozen (frozen feature extraction)", BN_ABLATION_NOTE))


if __name__ == "__main__":
    main()
