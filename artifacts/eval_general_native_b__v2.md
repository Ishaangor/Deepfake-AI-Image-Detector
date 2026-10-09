# general_native_b on CompEval-v2 subset (T=1.618, 3784 images)

| condition | AUC | bal.acc | fake P | fake R | fake F1 | real spec. | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| blur1 | 0.881 | 0.793 | 0.872 | 0.763 | 0.814 | 0.823 | 0.146 | 0.032 |
| clean | 0.909 | 0.807 | 0.911 | 0.725 | 0.807 | 0.888 | 0.141 | 0.022 |
| gray | 0.926 | 0.832 | 0.933 | 0.748 | 0.830 | 0.915 | 0.133 | 0.044 |
| half | 0.842 | 0.770 | 0.858 | 0.731 | 0.789 | 0.809 | 0.169 | 0.043 |
| jpeg50 | 0.954 | 0.882 | 0.945 | 0.842 | 0.891 | 0.923 | 0.105 | 0.094 |
| jpeg75 | 0.938 | 0.869 | 0.916 | 0.862 | 0.888 | 0.875 | 0.107 | 0.082 |
| quarter | 0.763 | 0.711 | 0.785 | 0.742 | 0.763 | 0.679 | 0.204 | 0.094 |
| social | 0.941 | 0.867 | 0.915 | 0.860 | 0.887 | 0.875 | 0.106 | 0.081 |

clean 95% bootstrap CI: auc [0.900, 0.918], balanced_acc [0.794, 0.819], fake_recall [0.707, 0.743], real_specificity [0.872, 0.903], fake_f1 [0.794, 0.820]

## Per generator (fake recall, clean)

| generator | n | recall | mean P(real) |
|---|---|---|---|
| Imagen3 | 252 | 0.50 | 0.48 |
| MidjourneyV6_1 | 249 | 0.53 | 0.49 |
| MidjourneyV5_2 | 350 | 0.56 | 0.46 |
| FLUX-schnell | 302 | 0.73 | 0.35 |
| kvikontent_midjourney_v6 | 257 | 0.74 | 0.34 |
| FLUX-dev | 339 | 0.79 | 0.31 |
| kandinsky_2_2 | 255 | 0.95 | 0.18 |
| stable_cascade | 311 | 0.98 | 0.16 |

## Per real source (specificity, clean)

| source | n | specificity | mean P(real) |
|---|---|---|---|
| LAION | 867 | 0.88 | 0.80 |
| RAISE | 126 | 0.97 | 0.79 |
| coco | 126 | 0.91 | 0.77 |
| ffhq | 126 | 1.00 | 0.90 |
| imagenet | 224 | 0.79 | 0.69 |
