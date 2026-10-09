# general_native_a on CompEval-v1 subset (T=1.396, 2162 images)

| condition | AUC | bal.acc | fake P | fake R | fake F1 | real spec. | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| blur1 | 0.654 | 0.606 | 0.691 | 0.600 | 0.642 | 0.612 | 0.265 | 0.174 |
| clean | 0.688 | 0.631 | 0.714 | 0.624 | 0.666 | 0.638 | 0.258 | 0.173 |
| gray | 0.677 | 0.634 | 0.711 | 0.654 | 0.681 | 0.614 | 0.253 | 0.159 |
| half | 0.671 | 0.627 | 0.690 | 0.731 | 0.710 | 0.523 | 0.250 | 0.162 |
| jpeg50 | 0.737 | 0.678 | 0.753 | 0.679 | 0.714 | 0.677 | 0.218 | 0.100 |
| jpeg75 | 0.712 | 0.662 | 0.731 | 0.697 | 0.714 | 0.627 | 0.229 | 0.122 |
| quarter | 0.632 | 0.593 | 0.652 | 0.822 | 0.727 | 0.365 | 0.271 | 0.195 |
| social | 0.728 | 0.672 | 0.736 | 0.717 | 0.726 | 0.627 | 0.218 | 0.103 |

clean 95% bootstrap CI: auc [0.667, 0.709], balanced_acc [0.614, 0.652], fake_recall [0.596, 0.654], real_specificity [0.607, 0.673], fake_f1 [0.645, 0.687]

## Per generator (fake recall, clean)

| generator | n | recall | mean P(real) |
|---|---|---|---|
| DFGAN | 75 | 0.13 | 0.77 |
| LCM_lora_sdxl | 72 | 0.18 | 0.73 |
| Dalle2 | 51 | 0.25 | 0.68 |
| Firefly_Image2 | 68 | 0.40 | 0.55 |
| LCM_lora_ssd1b | 75 | 0.55 | 0.48 |
| MidjourneyV6_1 | 75 | 0.55 | 0.46 |
| Firefly_Image3 | 51 | 0.55 | 0.45 |
| GALIP | 51 | 0.57 | 0.47 |
| MidjourneyV5_2 | 75 | 0.59 | 0.44 |
| IdeogramV2 | 51 | 0.59 | 0.47 |
| IdeogramV1 | 75 | 0.64 | 0.39 |
| Hourglass | 57 | 0.65 | 0.38 |
| kvikontent_midjourney_v6 | 75 | 0.68 | 0.36 |
| Imagen3 | 75 | 0.69 | 0.37 |
| Dalle3 | 51 | 0.86 | 0.16 |
| LCM_lora_sdv15 | 27 | 0.89 | 0.23 |
| FLUX-schnell | 13 | 0.92 | 0.23 |
| FLUX-dev | 75 | 0.93 | 0.15 |
| DeciDiffusionV2 | 51 | 0.98 | 0.08 |
| kandinsky_2_2 | 61 | 0.98 | 0.15 |
| stable_cascade | 75 | 0.99 | 0.12 |

## Per real source (specificity, clean)

| source | n | specificity | mean P(real) |
|---|---|---|---|
| LAION | 416 | 0.38 | 0.41 |
| RAISE | 100 | 1.00 | 0.91 |
| coco | 126 | 0.90 | 0.84 |
| ffhq | 114 | 0.99 | 0.92 |
| imagenet | 127 | 0.60 | 0.57 |
