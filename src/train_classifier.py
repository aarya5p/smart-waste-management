

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Fixed class list (CLAUDE.md). Order defines the label indices everywhere,
# so training and evaluation can never disagree about which index is which class.
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- reproducibility
def set_seed(seed: int):
    """Seed every source of randomness we can reach (rule 5)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Each DataLoader worker gets its own deterministic seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- data
class TrashDataset(Dataset):
    """Reads one of the split CSVs (columns: filepath,label). Paths are relative to data_root."""

    def __init__(self, csv_path, data_root, transform):
        self.df = pd.read_csv(csv_path)
        missing = set(self.df["label"]) - set(CLASSES)
        if missing:
            raise SystemExit(f"{csv_path} contains unexpected labels: {sorted(missing)}")
        self.data_root = Path(data_root)
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.data_root / row["filepath"]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.class_to_idx[row["label"]], row["filepath"]


def build_transforms(img_size: int):
    """Train gets augmentation; val/test only resize + normalize (rule 4)."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def class_weights_from_csv(csv_path) -> torch.Tensor:
    """Inverse-frequency weights: w_c = N / (K * n_c). Computed from TRAIN counts only."""
    counts = pd.read_csv(csv_path)["label"].value_counts()
    n = counts.sum()
    k = len(CLASSES)
    return torch.tensor([n / (k * counts[c]) for c in CLASSES], dtype=torch.float32)


# ---------------------------------------------------------------- model
# For each architecture: how to build it, where the classifier head lives, and which
# trailing blocks get unfrozen in finetune mode.
MODEL_REGISTRY = {
    "mobilenet_v2": {
        "ctor": models.mobilenet_v2,
        "head": lambda m: ("classifier.1", m.classifier[1]),
        "set_head": lambda m, layer: m.classifier.__setitem__(1, layer),
        "finetune_blocks": lambda m: [m.features[-4:]],
    },
    "mobilenet_v3_large": {
        "ctor": models.mobilenet_v3_large,
        "head": lambda m: ("classifier.3", m.classifier[3]),
        "set_head": lambda m, layer: m.classifier.__setitem__(3, layer),
        "finetune_blocks": lambda m: [m.features[-4:]],
    },
    "efficientnet_b0": {
        "ctor": models.efficientnet_b0,
        "head": lambda m: ("classifier.1", m.classifier[1]),
        "set_head": lambda m, layer: m.classifier.__setitem__(1, layer),
        "finetune_blocks": lambda m: [m.features[-3:]],
    },
    "resnet50": {
        "ctor": models.resnet50,
        "head": lambda m: ("fc", m.fc),
        "set_head": lambda m, layer: setattr(m, "fc", layer),
        "finetune_blocks": lambda m: [m.layer4],
    },
    "convnext_tiny": {
        "ctor": models.convnext_tiny,
        "head": lambda m: ("classifier.2", m.classifier[2]),
        "set_head": lambda m, layer: m.classifier.__setitem__(2, layer),
        "finetune_blocks": lambda m: [m.features[-2:]],
    },
}


def build_model(model_name: str, mode: str, num_classes: int = len(CLASSES), pretrained: bool = True):
    """
    Build an ImageNet-pretrained model with a fresh num_classes head.

    frozen   -> only the new head is trainable (pure feature extraction)
    finetune -> new head + the last blocks of the backbone are trainable
    Returns (model, head_module, backbone_trainable_modules).
    """
    if model_name not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]

    model = spec["ctor"](weights="IMAGENET1K_V1" if pretrained else None)

    # Swap the 1000-class ImageNet head for a fresh 6-class one.
    _, old_head = spec["head"](model)
    spec["set_head"](model, nn.Linear(old_head.in_features, num_classes))
    _, head_module = spec["head"](model)

    # Freeze everything, then re-enable exactly what this mode should train.
    for p in model.parameters():
        p.requires_grad = False
    for p in head_module.parameters():
        p.requires_grad = True

    backbone_blocks = []
    if mode == "finetune":
        backbone_blocks = spec["finetune_blocks"](model)
        for block in backbone_blocks:
            for p in block.parameters():
                p.requires_grad = True

    return model, head_module, backbone_blocks


# ---------------------------------------------------------------- train / eval loops
def run_epoch(model, loader, criterion, device, optimizer=None, limit_batches=0):
    """One pass over loader. Training if optimizer is given, else evaluation."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for i, (x, y, _) in enumerate(loader):
            if limit_batches and i >= limit_batches:
                break
            x, y = x.to(device), y.to(device)

            if is_train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1), seen


# ---------------------------------------------------------------- logging
def append_log_row(log_path: Path, date, model, settings, seed, accuracy, macro_f1, notes):
    """Append one row to LOG.md (rule 6), creating the table header if needed."""
    header = ("# Experiment Log\n\n"
              "| Date | Model | Settings | Seed | Accuracy | Macro F1 | Notes |\n"
              "|------|-------|----------|------|----------|----------|-------|\n")
    if not log_path.exists():
        log_path.write_text(header)
    with log_path.open("a") as f:
        f.write(f"| {date} | {model} | {settings} | {seed} | {accuracy} | {macro_f1} | {notes} |\n")


# ---------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description="Train a classifier on the 6 TrashNet classes")
    p.add_argument("--data_root", required=True, help="Folder the split CSV paths are relative to")
    p.add_argument("--train_csv", default="splits/train.csv")
    p.add_argument("--val_csv", default="splits/val.csv")
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--mode", required=True, choices=["frozen", "finetune"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None,
                   help="Head learning rate. Default: 1e-3 (frozen) / 1e-4 (finetune)")
    p.add_argument("--backbone_lr_mult", type=float, default=0.1,
                   help="finetune only: backbone lr = lr * this")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience on val loss")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--log_file", default="LOG.md")
    p.add_argument("--limit_batches", type=int, default=0,
                   help="Debug only: stop each epoch after N batches. Marks the run as a smoke test.")
    return p.parse_args()


def main():
    args = parse_args()
    smoke = args.limit_batches > 0
    if args.lr is None:
        args.lr = 1e-3 if args.mode == "frozen" else 1e-4

    set_seed(args.seed)
    device = pick_device(args.device)

    run_name = f"{args.model}_{args.mode}_seed{args.seed}" + ("_smoketest" if smoke else "")
    out_dir = Path(args.runs_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run: {run_name}\nDevice: {device}")
    if smoke:
        print(f"SMOKE TEST: only {args.limit_batches} batches/epoch, LOG.md will not be touched.")

    # ---- data (never re-split: we only read the CSVs, rule 1)
    train_tf, eval_tf = build_transforms(args.img_size)
    train_ds = TrashDataset(args.train_csv, args.data_root, train_tf)
    val_ds = TrashDataset(args.val_csv, args.data_root, eval_tf)

    g = torch.Generator()
    g.manual_seed(args.seed)
    common = dict(num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    print(f"train images: {len(train_ds)} | val images: {len(val_ds)}")

    # ---- model
    model, head_module, backbone_blocks = build_model(args.model, args.mode)
    model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {n_total:,} total | {n_train:,} trainable ({args.mode})")

    # ---- loss with class weights from the TRAIN split only
    weights = class_weights_from_csv(args.train_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("class weights: " + ", ".join(f"{c}={w:.3f}" for c, w in zip(CLASSES, weights.tolist())))

    # ---- optimizer: finetune uses a lower lr for the pretrained backbone
    head_params = list(head_module.parameters())
    if args.mode == "finetune":
        backbone_params = [p for b in backbone_blocks for p in b.parameters() if p.requires_grad]
        param_groups = [
            {"params": head_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        ]
    else:
        param_groups = [{"params": head_params, "lr": args.lr}]
    optimizer = torch.optim.Adam(param_groups)

    config = {
        **vars(args), "run_name": run_name, "device": str(device), "classes": CLASSES,
        "params_total": n_total, "params_trainable": n_train,
        "class_weights": dict(zip(CLASSES, weights.tolist())), "smoke_test": smoke,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ---- training loop with early stopping on VAL loss (rule 2)
    history, best_val_loss, best_epoch, epochs_no_improve = [], float("inf"), -1, 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, criterion, device,
                                       optimizer=optimizer, limit_batches=args.limit_batches)
        va_loss, va_acc, _ = run_epoch(model, val_loader, criterion, device,
                                       limit_batches=args.limit_batches)
        dt = time.time() - t0

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "seconds": dt})
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s", end="")

        if va_loss < best_val_loss:
            best_val_loss, best_epoch, epochs_no_improve = va_loss, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": args.model, "mode": args.mode, "seed": args.seed,
                "img_size": args.img_size, "classes": CLASSES, "epoch": epoch,
                "val_loss": va_loss, "val_acc": va_acc, "run_name": run_name,
            }, out_dir / "best.pt")
            print("  <- best, saved")
        else:
            epochs_no_improve += 1
            print(f"  (no improve {epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    total_time = time.time() - start
    best = next(h for h in history if h["epoch"] == best_epoch)
    summary = {
        "run_name": run_name, "best_epoch": best_epoch, "epochs_run": len(history),
        "best_val_loss": best_val_loss, "best_val_acc": best["val_acc"],
        "total_seconds": total_time, "history": history, "config": config,
    }
    (out_dir / "history.json").write_text(json.dumps(summary, indent=2))

    print(f"\nBest epoch {best_epoch}: val loss {best_val_loss:.4f}, val acc {best['val_acc']:.4f}")
    print(f"Checkpoint: {out_dir/'best.pt'}  |  History: {out_dir/'history.json'}")
    print(f"Total time: {total_time/60:.1f} min")

    # ---- LOG.md (rule 6). Skipped for smoke tests so the real log stays clean.
    if smoke:
        print("Smoke test: LOG.md not updated.")
    else:
        settings = (f"{args.mode}, {args.epochs}ep(max), bs{args.batch_size}, lr{args.lr:g}, "
                    f"img{args.img_size}, Adam, class-weighted CE, early-stop p{args.patience}")
        append_log_row(
            Path(args.log_file), datetime.now().strftime("%Y-%m-%d"), args.model, settings,
            args.seed, f"{best['val_acc']:.4f} (val)", "- (see evaluate.py)",
            f"train | best epoch {best_epoch}/{len(history)} | val loss {best_val_loss:.4f} "
            f"| {total_time/60:.1f} min on {device} | {run_name}",
        )
        print(f"Appended a row to {args.log_file}")


if __name__ == "__main__":
    main()
