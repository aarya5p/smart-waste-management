

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Summarize TACO categories")
    p.add_argument("--annotations", default="data/raw/taco/data/annotations.json")
    p.add_argument("--out", default="docs/taco_categories.csv")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.annotations, "r") as f:
        data = json.load(f)

    categories = {c["id"]: c for c in data["categories"]}

    ann_count = defaultdict(int)
    images_per_cat = defaultdict(set)
    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        ann_count[cat_id] += 1
        images_per_cat[cat_id].add(ann["image_id"])

    rows = []
    for cat_id, cat in categories.items():
        rows.append({
            "category_id": cat_id,
            "supercategory": cat["supercategory"],
            "name": cat["name"],
            "annotation_count": ann_count.get(cat_id, 0),
            "image_count": len(images_per_cat.get(cat_id, set())),
        })

    table = pd.DataFrame(rows).sort_values("annotation_count", ascending=False).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    print(table.to_string(index=False))
    print(f"\n{len(table)} categories | {table['annotation_count'].sum()} annotations total")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
