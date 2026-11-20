# Frozen classifiers: mean ± std over seeds (%)

Std = sample std (ddof=1). Models ordered by mean val macro F1. Val is for decisions; test is for reporting only.

Freeze policy: `bn_adapted` = pre-fix code, backbone weights frozen but BatchNorm running statistics adapted to TrashNet (ConvNeXt has LayerNorm, so it was truly frozen apart from stochastic depth); `frozen_parts_eval` = fixed code, frozen parts kept in eval mode.

## val

| Model | Seeds | Freeze policy | Accuracy | Macro F1 |
|---|---|---|---|---|
| convnext_tiny | 42,56,99 | bn_adapted | 93.05 ± 0.53 | 91.67 ± 0.70 |
| mobilenet_v2 | 42,56,99 | bn_adapted | 83.51 ± 1.01 | 81.58 ± 1.24 |
| mobilenet_v3_large | 42,56,99 | bn_adapted | 82.71 ± 1.01 | 79.43 ± 1.22 |
| efficientnet_b0 | 42,56,99 | bn_adapted | 81.73 ± 1.08 | 79.29 ± 1.26 |
| resnet50 | 42,56,99 | bn_adapted | 81.19 ± 0.41 | 77.68 ± 1.73 |

## test

| Model | Seeds | Freeze policy | Accuracy | Macro F1 |
|---|---|---|---|---|
| convnext_tiny | 42,56,99 | bn_adapted | 92.53 ± 0.53 | 91.48 ± 0.72 |
| mobilenet_v2 | 42,56,99 | bn_adapted | 81.69 ± 0.67 | 79.68 ± 1.13 |
| mobilenet_v3_large | 42,56,99 | bn_adapted | 84.18 ± 1.11 | 82.22 ± 1.45 |
| efficientnet_b0 | 42,56,99 | bn_adapted | 84.09 ± 1.32 | 82.18 ± 1.20 |
| resnet50 | 42,56,99 | bn_adapted | 79.47 ± 1.85 | 76.73 ± 2.15 |

