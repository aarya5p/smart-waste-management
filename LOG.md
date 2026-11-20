# Experiment Log

| Date | Run | Type | Model | Mode | Split | Seed | Accuracy | Macro F1 | Params | Settings | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-12 | mobilenet_v2_frozen_seed42 | train | mobilenet_v2 | frozen | val | 42 | 0.8235 | - | 2,231,558 | frozen, 30ep(max), bs32, lr0.001, img224, Adam, class-weighted CE, early-stop p5 | best epoch 19/24; val loss 0.5315; 5.6 min on cuda (Colab) |
| 2026-10-12 | mobilenet_v2_frozen_seed42 | eval | mobilenet_v2 | frozen | val | 42 | 0.8235 | 0.8015 | 2,231,558 | eval on splits/val.csv, img224 | macro P 0.7983, R 0.8220; ckpt 8.73 MB; GMACs 0.31; 5.1 ms/img on cuda (laptop, not for report) |
| 2026-10-12 | mobilenet_v2_frozen_seed42 | eval | mobilenet_v2 | frozen | test | 42 | 0.8107 | 0.7842 | 2,231,558 | eval on splits/test.csv, img224 | macro P 0.7811, R 0.8044; ckpt 8.73 MB; GMACs 0.31; 5.3 ms/img on cuda (laptop, not for report) |
