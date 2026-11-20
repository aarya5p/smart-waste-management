"""Convert TACO (COCO format) to two YOLO detection datasets that share one image-level split.

  data/processed/taco_yolo_6class/  boxes labelled with the 6 TrashNet classes (same order and ids as
                                    the classifiers: cardboard, glass, metal, paper, plastic, trash)
  data/processed/taco_yolo_1class/  same images, same boxes, every box labelled "waste"

Steps
  1. Map every TACO category to a class with docs/taco_mapping.csv (approved). Images that contain an
     EXCLUDE object (Battery) are dropped entirely, so no real object is left unlabelled.
  2. Split by IMAGE 70/15/15. Near-duplicate photos (pHash distance <= 5, same threshold as
     prepare_trashnet.py) are grouped and always go to the same split (rule 3).
     Seed selection (label balance): every seed in --seed_min..--seed_max is tried. For each class,
     val share = val boxes / all boxes of that class, same for test; the seed's score is the LARGEST
     |share - 15%| over all 6 classes and both splits. The seed with the smallest score wins (ties ->
     lowest seed), among seeds where every class has >= --min_boxes boxes in val and in test.
     This uses label COUNTS only, never any model result, and was done before any training, so it
     does not leak test information (rule 2). Seed, criterion and per-class table are saved.
  3. Save the split as splits/taco_{train,val,test}.txt (committed). If those files already exist they
     are READ, not regenerated (rule 1: never re-split) - pass --resplit to override deliberately.
  4. Images are re-saved with their EXIF rotation applied (152 TACO photos are stored sideways with
     an EXIF flag, and TACO's boxes are in the rotated, as-displayed coordinates), EXIF stripped, and
     downscaled so the longest side is <= --max_side. YOLO coordinates are normalised, so resizing
     does not change the labels. The 1-class dataset hard-links the same image files.
  5. Prints per-class box counts per split, draws the boxes on --n_check random images into
     results/taco_check/, and (with --zip) writes data/processed/taco_yolo.zip for uploading to Drive
     (both datasets, images stored once).

Run:  python src/taco_to_yolo.py --zip
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_classifier import CLASSES  # noqa: E402  single source of truth for class order

SPLITS = ["train", "val", "test"]
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
COLORS = {"cardboard": (166, 118, 29), "glass": (31, 158, 137), "metal": (120, 120, 140),
          "paper": (58, 120, 216), "plastic": (220, 50, 120), "trash": (40, 40, 40)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--taco_dir", default="data/raw/taco/data")
    p.add_argument("--mapping", default="docs/taco_mapping.csv")
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--seed_min", type=int, default=0, help="seed search range (inclusive)")
    p.add_argument("--seed_max", type=int, default=999)
    p.add_argument("--min_boxes", type=int, default=10, help="hard constraint: per class, in val and in test")
    p.add_argument("--check_seed", type=int, default=42, help="which random images get box-check drawings")
    p.add_argument("--phash_threshold", type=int, default=5)
    p.add_argument("--max_side", type=int, default=1280)
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--check_dir", default="results/taco_check")
    p.add_argument("--n_check", type=int, default=5)
    p.add_argument("--resplit", action="store_true", help="ignore existing splits/taco_*.txt")
    p.add_argument("--zip", action="store_true",
                   help="also write data/processed/taco_yolo.zip (both datasets, images stored once) for Drive")
    return p.parse_args()


# ---------------------------------------------------------------- annotations
def load_mapping(path):
    rows = list(csv.DictReader(open(path)))
    mapping = {r["taco_category"]: r["mapped_class"] for r in rows}
    bad = {c for c in mapping.values() if c not in CLASSES and c != "EXCLUDE"}
    if bad:
        raise SystemExit(f"{path}: unknown mapped_class values {bad}")
    return mapping


def load_boxes(taco_dir, mapping):
    """-> (images {file_name: (w, h)}, boxes {file_name: [(class, x, y, w, h), ...]}, report)"""
    coco = json.loads((taco_dir / "annotations.json").read_text())
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    missing = set(cat_name.values()) - set(mapping)
    if missing:
        raise SystemExit(f"Categories missing from the mapping: {sorted(missing)}")
    images = {i["file_name"]: (i["width"], i["height"]) for i in coco["images"]}
    by_id = {i["id"]: i["file_name"] for i in coco["images"]}

    boxes, excluded_imgs, n_clipped, n_degenerate = defaultdict(list), set(), 0, 0
    for a in coco["annotations"]:
        fn = by_id[a["image_id"]]
        cls = mapping[cat_name[a["category_id"]]]
        if cls == "EXCLUDE":
            excluded_imgs.add(fn)
            continue
        W, H = images[fn]
        x, y, w, h = a["bbox"]
        x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(W, x + w), min(H, y + h)
        if (x0, y0, x1, y1) != (x, y, x + w, y + h):
            n_clipped += 1
        if x1 - x0 < 1 or y1 - y0 < 1:
            n_degenerate += 1
            continue
        boxes[fn].append((cls, x0, y0, x1 - x0, y1 - y0))

    for fn in excluded_imgs:        # drop the whole image: its EXCLUDE object would be unlabelled
        boxes.pop(fn, None)
    kept = {fn: images[fn] for fn in boxes}
    report = {"images_in_annotations": len(images), "images_dropped_exclude": sorted(excluded_imgs),
              "images_without_boxes_after_mapping": sorted(set(images) - set(kept) - excluded_imgs),
              "boxes_clipped_to_image": n_clipped, "boxes_dropped_degenerate": n_degenerate}
    return kept, dict(boxes), report


# ---------------------------------------------------------------- split
def near_duplicate_groups(taco_dir, files, threshold):
    """Union-find over pHash distance <= threshold, computed on the EXIF-rotated image."""
    hashes = {}
    for fn in files:
        with Image.open(taco_dir / fn) as im:
            im.draft("RGB", (512, 512))          # fast JPEG decode at reduced size
            hashes[fn] = imagehash.phash(ImageOps.exif_transpose(im))
    parent = {f: f for f in files}

    def find(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    pairs = []
    for i, a in enumerate(files):
        for b in files[i + 1:]:
            d = hashes[a] - hashes[b]
            if d <= threshold:
                pairs.append((a, b, int(d)))
                parent[find(a)] = find(b)
    groups = defaultdict(list)
    for f in files:
        groups[find(f)].append(f)
    return sorted(sorted(g) for g in groups.values()), pairs


def split_groups(groups, seed):
    """Shuffle groups, then fill train/val/test up to 70/15/15 of the IMAGE count."""
    g = list(groups)
    random.Random(seed).shuffle(g)
    n = sum(len(x) for x in g)
    target_train, target_val = round(n * FRACTIONS["train"]), round(n * FRACTIONS["val"])
    out, count = {s: [] for s in SPLITS}, 0
    for grp in g:
        s = "train" if count < target_train else "val" if count < target_train + target_val else "test"
        out[s] += grp
        count += len(grp)
    return {s: sorted(v) for s, v in out.items()}


def box_counts(split, boxes):
    return {s: Counter(c for fn in split[s] for c, *_ in boxes[fn]) for s in SPLITS}


def min_ok(counts, k):
    return all(counts[s][c] >= k for s in ("val", "test") for c in CLASSES)


def balance_score(counts):
    """Largest |share - 15%| over the 6 classes x {val, test}; share = split boxes / class boxes."""
    worst = 0.0
    for c in CLASSES:
        tot = sum(counts[s][c] for s in SPLITS)
        for s in ("val", "test"):
            worst = max(worst, abs(counts[s][c] / tot - FRACTIONS[s]))
    return worst


def class_table(counts, split):
    """Per-class boxes and shares per split, plus image counts - stored in taco_split_info.json."""
    t = {}
    for c in CLASSES:
        tot = sum(counts[s][c] for s in SPLITS)
        t[c] = {**{s: counts[s][c] for s in SPLITS}, "total": tot,
                **{f"{s}_share": round(counts[s][c] / tot, 4) for s in SPLITS}}
    t["_images"] = {s: len(split[s]) for s in SPLITS}
    return t


# ---------------------------------------------------------------- writing
def out_name(fn):
    return fn.replace("/", "_").rsplit(".", 1)[0] + ".jpg"      # batch_1/000006.JPG -> batch_1_000006.jpg


def yolo_line(cls_id, box, W, H):
    _, x, y, w, h = box
    return f"{cls_id} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"


def write_datasets(taco_dir, out_dir, split, images, boxes, max_side, quality):
    d6, d1 = out_dir / "taco_yolo_6class", out_dir / "taco_yolo_1class"
    for d in (d6, d1):
        shutil.rmtree(d, ignore_errors=True)        # generated data only; raw TACO is never touched
        for s in SPLITS:
            (d / "images" / s).mkdir(parents=True)
            (d / "labels" / s).mkdir(parents=True)
    for s in SPLITS:
        for fn in split[s]:
            W, H = images[fn]
            with Image.open(taco_dir / fn) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
            if im.size != (W, H):
                raise SystemExit(f"{fn}: size after EXIF rotation {im.size} != annotation size {(W, H)}")
            scale = min(1.0, max_side / max(W, H))
            if scale < 1:
                im = im.resize((round(W * scale), round(H * scale)), Image.LANCZOS)
            name = out_name(fn)
            im.save(d6 / "images" / s / name, quality=quality)        # no exif= -> EXIF stripped
            try:
                os.link(d6 / "images" / s / name, d1 / "images" / s / name)
            except OSError:
                shutil.copy2(d6 / "images" / s / name, d1 / "images" / s / name)
            stem = name[:-4]
            (d6 / "labels" / s / f"{stem}.txt").write_text(
                "\n".join(yolo_line(CLASSES.index(b[0]), b, W, H) for b in boxes[fn]) + "\n")
            (d1 / "labels" / s / f"{stem}.txt").write_text(
                "\n".join(yolo_line(0, b, W, H) for b in boxes[fn]) + "\n")
    for d, names in ((d6, CLASSES), (d1, ["waste"])):
        # absolute path for local use; the Colab notebook rewrites this file with its own path
        (d / "data.yaml").write_text(
            f"# generated by src/taco_to_yolo.py\npath: {d.resolve()}\ntrain: images/train\n"
            f"val: images/val\ntest: images/test\nnc: {len(names)}\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    return d6, d1


def draw_checks(d6, n, seed, check_dir):
    """Draw boxes from the WRITTEN 6-class labels on n random images (verifies the final files)."""
    shutil.rmtree(check_dir, ignore_errors=True)
    check_dir.mkdir(parents=True)
    imgs = sorted((d6 / "images").rglob("*.jpg"))
    picks = random.Random(seed).sample(imgs, n)
    for p in picks:
        split = p.parent.name
        im = Image.open(p).convert("RGB")
        W, H = im.size
        dr = ImageDraw.Draw(im)
        lw = max(2, W // 300)
        for row in (d6 / "labels" / split / f"{p.stem}.txt").read_text().strip().splitlines():
            c, xc, yc, w, h = row.split()
            c, xc, yc, w, h = int(c), float(xc) * W, float(yc) * H, float(w) * W, float(h) * H
            box = [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]
            col = COLORS[CLASSES[c]]
            dr.rectangle(box, outline=col, width=lw)
            tb = dr.textbbox((box[0], box[1]), CLASSES[c])
            dr.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=col)
            dr.text((box[0], box[1]), CLASSES[c], fill=(255, 255, 255))
        im.thumbnail((1024, 1024))
        im.save(check_dir / f"{split}_{p.name}", quality=90)
    return picks


def zip_datasets(d6, d1):
    """ONE zip for Drive: the 6-class dataset complete + the 1-class labels and data.yaml.
    The 1-class images are identical files, so they are not stored twice; the Colab notebook
    recreates taco_yolo_1class/images with hard links (cp -al)."""
    z = d6.parent / "taco_yolo.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:      # JPEGs are already compressed
        for f in sorted(d6.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(d6.parent))
        for f in sorted(d1.rglob("*")):
            if f.is_file() and "images" not in f.relative_to(d1).parts:
                zf.write(f, f.relative_to(d1.parent))
    return z


# ---------------------------------------------------------------- main
def main():
    a = parse_args()
    taco_dir, out_dir, splits_dir = Path(a.taco_dir), Path(a.out_dir), Path(a.splits_dir)
    images, boxes, report = load_boxes(taco_dir, load_mapping(a.mapping))
    files = sorted(images)
    print(f"{report['images_in_annotations']} images in annotations.json; {len(files)} kept "
          f"({len(report['images_dropped_exclude'])} dropped for containing an EXCLUDE object: "
          f"{report['images_dropped_exclude']}); {report['boxes_clipped_to_image']} box(es) clipped "
          f"to the image, {report['boxes_dropped_degenerate']} degenerate dropped")
    missing = [f for f in files if not (taco_dir / f).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing on disk, e.g. {missing[:3]}")

    split_files = {s: splits_dir / f"taco_{s}.txt" for s in SPLITS}
    info_path = splits_dir / "taco_split_info.json"
    if all(p.exists() for p in split_files.values()) and not a.resplit:
        print("Using the EXISTING split in splits/taco_*.txt (rule 1: never re-split).")
        split = {s: split_files[s].read_text().split() for s in SPLITS}
        if sorted(sum(split.values(), [])) != files:
            raise SystemExit("splits/taco_*.txt don't match the current kept image set - investigate "
                             "before using --resplit.")
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
    else:
        print(f"Grouping near-duplicates (pHash distance <= {a.phash_threshold}) ...")
        groups, pairs = near_duplicate_groups(taco_dir, files, a.phash_threshold)
        print(f"  {len(pairs)} near-duplicate pair(s): {pairs}")
        results = []
        for seed in range(a.seed_min, a.seed_max + 1):
            c = box_counts(split_groups(groups, seed), boxes)
            results.append((seed, balance_score(c), min_ok(c, a.min_boxes)))
        eligible = [r for r in results if r[2]]
        if not eligible:
            raise SystemExit(f"No seed in {a.seed_min}..{a.seed_max} has >= {a.min_boxes} boxes per class "
                             "in val and test")
        seed, score, _ = min(eligible, key=lambda r: (r[1], r[0]))
        split = split_groups(groups, seed)
        scores = sorted(r[1] for r in eligible)
        print(f"Tried seeds {a.seed_min}-{a.seed_max}: {len(eligible)} meet the >= {a.min_boxes}-box "
              f"constraint; best max deviation {score:.4f} (seed {seed}); median {scores[len(scores)//2]:.4f}")
        for s in SPLITS:
            split_files[s].write_text("\n".join(split[s]) + "\n")
        counts = box_counts(split, boxes)
        old42 = next((r for r in results if r[0] == 42), None)
        info = {
            "seed_used": seed,
            "selection": {
                "criterion": "minimise the largest absolute deviation, over all 6 classes, of each "
                             "class's val share and test share (boxes in split / boxes of that class) "
                             "from 15%",
                "constraint": f"every class >= {a.min_boxes} boxes in val and in test",
                "seeds_tried": [a.seed_min, a.seed_max], "n_seeds_meeting_constraint": len(eligible),
                "best_max_deviation": round(score, 4),
                "median_max_deviation_over_eligible_seeds": round(scores[len(scores) // 2], 4),
                "uses": "label counts only - chosen before any model was trained (rule 2)",
                "superseded": None if old42 is None else
                    f"an earlier split with seed 42 (max deviation {old42[1]:.4f}; glass val/test "
                    "17/71 boxes) was replaced by this one on 2026-11-20, before any training",
            },
            "per_class": class_table(counts, split),
            "fractions": FRACTIONS, "unit": "image (near-duplicate groups kept together)",
            "phash_threshold": a.phash_threshold, "near_duplicate_pairs": pairs, **report,
            "frozen": "do not re-split; src/taco_to_yolo.py reads these files unless --resplit is given",
        }
        info_path.write_text(json.dumps(info, indent=2) + "\n")

    counts = box_counts(split, boxes)
    if not min_ok(counts, a.min_boxes):
        print(f"WARNING: some class has < {a.min_boxes} boxes in val or test")

    print(f"\nWriting datasets (EXIF applied, max side {a.max_side}) ...")
    d6, d1 = write_datasets(taco_dir, out_dir, split, images, boxes, a.max_side, a.jpeg_quality)

    # ---- per-class box counts per split
    print(f"\nSplit seed: {info.get('seed_used', '?')}")
    hdr = (f"{'class':10s}" + "".join(f"{s:>8s}" for s in SPLITS) + f"{'total':>8s}"
           + f"{'val %':>8s}{'test %':>8s}")
    print(hdr + "\n" + "-" * len(hdr))
    for c in CLASSES:
        tot_c = sum(counts[s][c] for s in SPLITS)
        print(f"{c:10s}" + "".join(f"{counts[s][c]:8d}" for s in SPLITS) + f"{tot_c:8d}"
              + f"{100 * counts['val'][c] / tot_c:8.1f}{100 * counts['test'][c] / tot_c:8.1f}")
    print("-" * len(hdr))
    tot = {s: sum(counts[s].values()) for s in SPLITS}
    print(f"{'all boxes':10s}" + "".join(f"{tot[s]:8d}" for s in SPLITS) + f"{sum(tot.values()):8d}"
          f"   (1-class dataset: same numbers, all 'waste')")
    print(f"{'images':10s}" + "".join(f"{len(split[s]):8d}" for s in SPLITS) + f"{len(files):8d}")

    picks = draw_checks(d6, a.n_check, a.check_seed, Path(a.check_dir))
    print(f"\nBox check images -> {a.check_dir}/ : {[p.parent.name + '/' + p.name for p in picks]}")

    if a.zip:
        z = zip_datasets(d6, d1)
        print(f"Zip for Drive: {z} ({z.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
