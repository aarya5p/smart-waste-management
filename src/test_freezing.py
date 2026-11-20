"""Unit test: does "frozen" really mean frozen?

For each of the 5 classifiers, in frozen and finetune mode, build the model exactly as
train_classifier.py does, run a few training steps on random data through the REAL training
code path (run_epoch + make_optimizer + set_train_mode), and check:

  1. every frozen parameter (requires_grad=False) is bit-for-bit unchanged
  2. every BatchNorm running_mean / running_var / num_batches_tracked in a frozen part is unchanged
  3. the head's weights changed (training actually happened)
  4. every module outside the trainable blocks was in eval mode during training
     (so stochastic depth / dropout inside the frozen backbone were off)
  finetune only:
  5. the unfrozen backbone parameters changed

Plus a CONTROL per model: the same steps with the old behaviour (plain model.train()) must
change at least one frozen BatchNorm buffer for the BN models - proving check 2 can actually
detect the bug it guards against. (ConvNeXt has no BatchNorm, so its control is skipped.)

Uses randomly initialised weights (pretrained=False) so it runs offline in ~1 min on CPU; the
freezing logic does not depend on the weight values.

Run:  python src/test_freezing.py
"""
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_classifier import (CLASSES, MODEL_REGISTRY, build_model, head_block,  # noqa: E402
                              make_optimizer, run_epoch)

MODELS = ["mobilenet_v2", "mobilenet_v3_large", "efficientnet_b0", "resnet50", "convnext_tiny"]
STEPS, BATCH, IMG = 3, 4, 96


def random_loader(seed):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(BATCH, 3, IMG, IMG, generator=g),
             torch.randint(0, len(CLASSES), (BATCH,), generator=g),
             ["x"] * BATCH) for _ in range(STEPS)]


def module_ids(modules):
    return {id(m) for root in modules for m in root.modules()}


def frozen_bn_modules(model):
    """BatchNorm layers whose parameters are all frozen."""
    return {name: m for name, m in model.named_modules()
            if isinstance(m, _BatchNorm) and not any(p.requires_grad for p in m.parameters())}


def bn_buffers(bn_dict):
    return {f"{n}.{b}": getattr(m, b).detach().clone()
            for n, m in bn_dict.items() for b in ("running_mean", "running_var", "num_batches_tracked")
            if getattr(m, b, None) is not None}


def train_steps(model, mode, head, blocks, train_modules, loader, lr):
    opt = make_optimizer(mode, lr, 0.1, head, blocks)
    run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"),
              optimizer=opt, train_modules=train_modules)


def check(model_name, mode):
    torch.manual_seed(0)
    model, head, blocks = build_model(model_name, mode, pretrained=False)
    train_modules = [head_block(model_name, model), *blocks]
    lr = 1e-3 if mode == "frozen" else 1e-4

    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    head_names = {n for n, p in model.named_parameters() if any(p is q for q in head.parameters())}
    fbn = frozen_bn_modules(model)
    bn_before = bn_buffers(fbn)

    train_steps(model, mode, head, blocks, train_modules, random_loader(1), lr)

    errors, after = [], dict(model.named_parameters())
    # 1. frozen params unchanged
    changed_frozen = [n for n in frozen_names if not torch.equal(after[n], params_before[n])]
    if changed_frozen:
        errors.append(f"{len(changed_frozen)} frozen params changed, e.g. {changed_frozen[:3]}")
    # 2. frozen BN buffers unchanged
    bn_after = bn_buffers(fbn)
    changed_bn = [k for k in bn_before if not torch.equal(bn_before[k], bn_after[k])]
    if changed_bn:
        errors.append(f"{len(changed_bn)} frozen BN buffers changed, e.g. {changed_bn[:3]}")
    # 3. head changed
    if all(torch.equal(after[n], params_before[n]) for n in head_names):
        errors.append("head weights did not change")
    # 4. everything outside train_modules was in eval mode during training
    allowed = module_ids(train_modules)
    in_train = [n for n, m in model.named_modules() if m.training and id(m) not in allowed]
    if in_train:
        errors.append(f"{len(in_train)} frozen modules were in train mode, e.g. {in_train[:3]}")
    # 5. finetune: unfrozen backbone moved
    backbone_trainable = [n for n in trainable_names if n not in head_names]
    n_bb_changed = sum(not torch.equal(after[n], params_before[n]) for n in backbone_trainable)
    if mode == "finetune" and n_bb_changed == 0:
        errors.append("no unfrozen backbone parameter changed")
    if mode == "frozen" and backbone_trainable:
        errors.append(f"frozen mode has trainable backbone params: {backbone_trainable[:3]}")

    n_tr = sum(after[n].numel() for n in trainable_names)
    n_all = sum(p.numel() for p in after.values())
    info = (f"trainable {n_tr:>10,}/{n_all:>10,} | frozen params {len(frozen_names):3d} unchanged | "
            f"frozen BN layers {len(fbn):3d} stats unchanged | head changed"
            + (f" | backbone tensors changed {n_bb_changed}/{len(backbone_trainable)}"
               if mode == "finetune" else ""))
    return errors, info, len(fbn)


def control_old_behaviour(model_name, mode):
    """Old code put the whole model in train mode. Frozen BN stats must then change."""
    torch.manual_seed(0)
    model, head, blocks = build_model(model_name, mode, pretrained=False)
    fbn = frozen_bn_modules(model)
    before = bn_buffers(fbn)
    train_steps(model, mode, head, blocks, [model], random_loader(1),   # [model] == model.train()
                1e-3 if mode == "frozen" else 1e-4)
    after = bn_buffers(fbn)
    return sum(not torch.equal(before[k], after[k]) for k in before)


def main():
    assert set(MODELS) == set(MODEL_REGISTRY), "test does not cover every registered model"
    failures = 0
    for mode in ["frozen", "finetune"]:
        print(f"\n===== mode: {mode} =====")
        for name in MODELS:
            errors, info, n_fbn = check(name, mode)
            status = "PASS" if not errors else "FAIL"
            print(f"[{status}] {name:20s} {info}")
            for e in errors:
                print(f"        - {e}")
            failures += bool(errors)
            if n_fbn:
                n = control_old_behaviour(name, mode)
                ok = n > 0
                print(f"        control (old model.train()): {n} frozen BN buffers changed "
                      f"-> {'detected, as expected' if ok else 'NOT detected - test is too weak!'}")
                failures += not ok
            else:
                print("        control skipped: no BatchNorm (LayerNorm has no running stats)")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
