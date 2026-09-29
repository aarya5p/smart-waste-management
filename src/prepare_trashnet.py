"""
prepare_trashnet.py
Clean, deduplicate and split the TrashNet dataset.

What it does
  1. Scans <data_dir>/<class_name>/*.jpg (TrashNet's "dataset-resized" layout).
  2. Drops unreadable / corrupt images.
  3. Finds near-duplicates with perceptual hashing (pHash + Hamming distance).
       - same-class duplicates  -> keep one image, drop the rest
       - cross-class duplicates -> drop ALL images in the group (label conflict)
  4. Saves side-by-side previews of every duplicate pair so you can check them.
  5. Makes a stratified 70/15/15 train/val/test split with a fixed seed.
  6. Writes train.csv, val.csv, test.csv (columns: filepath,label) with paths
     RELATIVE to data_dir, so the same CSVs work on your laptop and on Colab.

Usage
  python prepare_trashnet.py --data_dir data/raw/trashnet/dataset-resized --out_dir splits

  After looking at the previews in splits/duplicate_review/, if some flagged
  pairs are NOT real duplicates, list one of their file paths per line
  (relative paths, exactly as in duplicates.csv) in a text file and re-run:
  python prepare_trashnet.py ... --keep keep.txt

Requirements: pip install pillow imagehash numpy pandas scikit-learn
"""

import argparse
import json
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Clean, dedupe and split TrashNet")
    p.add_argument("--data_dir", required=True, help="Folder containing one subfolder per class")
    p.add_argument("--out_dir", default="splits", help="Where CSVs and reports are written")
    p.add_argument("--threshold", type=int, default=5,
                   help="Max Hamming distance between pHashes to count as duplicate (0-64). "
                        "Lower = stricter. 5 is a reasonable start.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--keep", default=None,
                   help="Optional text file of relative paths that must never be dropped")
    p.add_argument("--no_previews", action="store_true", help="Skip saving pair preview images")
    return p.parse_args()


# ---------------------------------------------------------------- scanning
def scan_images(data_dir: Path):
    records, corrupt = [], []
    class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not class_dirs:
        raise SystemExit(f"No class subfolders found in {data_dir}")
    for cdir in class_dirs:
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            rel = f.relative_to(data_dir).as_posix()
            try:
                with Image.open(f) as im:
                    im.verify()                     # catches truncated/corrupt files
                with Image.open(f) as im:
                    h = imagehash.phash(im.convert("RGB"))
                records.append({"filepath": rel, "label": cdir.name, "hash": h})
            except Exception as e:                  # noqa: BLE001
                corrupt.append({"filepath": rel, "error": str(e)})
    return records, corrupt


# ---------------------------------------------------------------- dedup
def hash_to_uint64(h: imagehash.ImageHash) -> np.uint64:
    bits = h.hash.flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def find_duplicate_pairs(records, threshold):
    """Return list of (i, j, distance) with i < j and distance <= threshold."""
    codes = np.array([hash_to_uint64(r["hash"]) for r in records], dtype=np.uint64)
    pairs = []
    for i in range(len(codes) - 1):
        x = np.bitwise_xor(codes[i + 1:], codes[i])
        dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for k in np.nonzero(dist <= threshold)[0]:
            pairs.append((i, i + 1 + int(k), int(dist[k])))
    return pairs


def group_pairs(n, pairs):
    """Union-find: merge pairs into groups of mutually connected duplicates."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    groups = {}
    members = {idx for p in pairs for idx in p[:2]}
    for m in members:
        groups.setdefault(find(m), []).append(m)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def decide_drops(records, groups, keep_set):
    drops, conflicts = {}, []
    for g in groups:
        labels = {records[i]["label"] for i in g}
        if len(labels) > 1:
            # Same picture filed under different classes: we can't trust either label.
            conflicts.append([records[i]["filepath"] for i in g])
            for i in g:
                if records[i]["filepath"] not in keep_set:
                    drops[i] = "cross-class duplicate (label conflict)"
        else:
            # Keep the first one (or any whitelisted one), drop the others.
            kept = next((i for i in g if records[i]["filepath"] in keep_set), g[0])
            for i in g:
                if i != kept and records[i]["filepath"] not in keep_set:
                    drops[i] = f"same-class duplicate of {records[kept]['filepath']}"
    return drops, conflicts


def save_preview(data_dir, a, b, dist, out_path):
    with Image.open(data_dir / a["filepath"]) as ia, Image.open(data_dir / b["filepath"]) as ib:
        ia, ib = ia.convert("RGB"), ib.convert("RGB")
        h = 256
        ia = ia.resize((int(ia.width * h / ia.height), h))
        ib = ib.resize((int(ib.width * h / ib.height), h))
        canvas = Image.new("RGB", (ia.width + ib.width + 10, h), "white")
        canvas.paste(ia, (0, 0))
        canvas.paste(ib, (ia.width + 10, 0))
        canvas.save(out_path, quality=85)


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_set = set()
    if args.keep:
        keep_set = {ln.strip() for ln in Path(args.keep).read_text().splitlines() if ln.strip()}

    print(f"Scanning {data_dir} ...")
    records, corrupt = scan_images(data_dir)
    print(f"  readable images: {len(records)} | corrupt/unreadable: {len(corrupt)}")

    print(f"Finding near-duplicates (pHash distance <= {args.threshold}) ...")
    pairs = find_duplicate_pairs(records, args.threshold)
    groups = group_pairs(len(records), pairs)
    drops, conflicts = decide_drops(records, groups, keep_set)
    print(f"  duplicate pairs: {len(pairs)} | groups: {len(groups)} | "
          f"cross-class groups: {len(conflicts)} | images dropped: {len(drops)}")

    # Report of every flagged pair
    dup_rows = [{
        "file_a": records[i]["filepath"], "label_a": records[i]["label"],
        "file_b": records[j]["filepath"], "label_b": records[j]["label"],
        "hamming_distance": d,
    } for i, j, d in pairs]
    pd.DataFrame(dup_rows, columns=["file_a", "label_a", "file_b", "label_b",
                                    "hamming_distance"]).to_csv(out_dir / "duplicates.csv", index=False)
    pd.DataFrame([{"filepath": records[i]["filepath"], "label": records[i]["label"], "reason": r}
                  for i, r in drops.items()],
                 columns=["filepath", "label", "reason"]).to_csv(out_dir / "removed.csv", index=False)
    pd.DataFrame(corrupt, columns=["filepath", "error"]).to_csv(out_dir / "corrupt.csv", index=False)

    if pairs and not args.no_previews:
        rev = out_dir / "duplicate_review"
        rev.mkdir(exist_ok=True)
        for n, (i, j, d) in enumerate(pairs):
            name = f"{n:04d}_d{d}_{records[i]['label']}_vs_{records[j]['label']}.jpg"
            save_preview(data_dir, records[i], records[j], d, rev / name)
        print(f"  previews saved to {rev}/ (check them!)")

    # Clean table
    clean = pd.DataFrame([{"filepath": r["filepath"], "label": r["label"]}
                          for k, r in enumerate(records) if k not in drops])

    # Stratified split: first carve off val+test, then split that in two
    holdout = args.val_frac + args.test_frac
    train, temp = train_test_split(clean, test_size=holdout, stratify=clean["label"],
                                   random_state=args.seed)
    val, test = train_test_split(temp, test_size=args.test_frac / holdout, stratify=temp["label"],
                                 random_state=args.seed)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.sort_values(["label", "filepath"]).to_csv(out_dir / f"{name}.csv", index=False)

    # Leakage check: no file may appear in more than one split
    sets = [set(train.filepath), set(val.filepath), set(test.filepath)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "Split overlap!"

    # Summary table
    table = pd.DataFrame({
        "original": pd.Series([r["label"] for r in records]).value_counts(),
        "removed": pd.Series([records[i]["label"] for i in drops], dtype=object).value_counts(),
        "train": train.label.value_counts(),
        "val": val.label.value_counts(),
        "test": test.label.value_counts(),
    }).fillna(0).astype(int)
    table.loc["TOTAL"] = table.sum()
    print("\nClass counts:\n" + table.to_string())
    table.to_csv(out_dir / "class_counts.csv")

    summary = {
        "data_dir": str(data_dir), "seed": args.seed, "threshold": args.threshold,
        "images_found": len(records) + len(corrupt), "corrupt": len(corrupt),
        "duplicate_pairs": len(pairs), "cross_class_groups": len(conflicts),
        "removed": len(drops), "final": len(clean),
        "train": len(train), "val": len(val), "test": len(test),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Files written to {out_dir}/")


if __name__ == "__main__":
    main()
