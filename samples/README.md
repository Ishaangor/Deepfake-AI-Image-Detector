# samples/

Eight demo images used by the web UI's *Try a demo image* gallery. `manifest.json` records each image's label, source,
licence, author and SHA-256 (checked by the tests).

**How they were chosen.** Drawn at random from held-out data, then filtered to avoid identifiable people. Failure cases
were kept on purpose. None was used for training. The set is a demonstration, **not** a measure of accuracy; see the
README for the benchmark.

| file | dataset label | source | model output (`general_native_b`) |
|---|---|---|---|
| `ai_stable_cascade_sheep.png` | AI-generated (Stable Cascade) | CompEval-v2 benchmark | likely AI, P(AI) 0.86 - correct |
| `ai_flux_dev_pattern.png` | AI-generated (FLUX-dev) | CompEval-v2 benchmark | likely AI, P(AI) 0.94 - correct |
| `ai_midjourney_plant.png` | AI-generated (Midjourney v6.1) | CompEval-v2 benchmark | inconclusive, P(AI) 0.45 |
| `ai_imagen3_city.png` | AI-generated (Imagen 3) | CompEval-v2 benchmark | likely real, P(AI) 0.11 - **missed** |
| `real_coffee_cup.jpg` | real photograph | Open Images V7 validation | likely real, P(AI) 0.04 - correct |
| `real_fish_dish.jpg` | real photograph | Open Images V7 validation | likely real, P(AI) 0.004 - correct |
| `real_sunflower.jpg` | real photograph | Open Images V7 validation | likely AI, P(AI) 0.82 - **false alarm** |
| `real_tasmanian_devils.jpg` | real photograph | Open Images V7 validation | inconclusive, P(AI) 0.21 |

Labels come from the datasets themselves and were cross-checked against the benchmark selection table
(`artifacts/ce_sel.parquet`, `label` column) and the Open Images cache (`split = val`).

## Licences and attribution
* **Real photographs** come from Open Images V7, Flickr images licensed **CC BY 2.0**
  (https://creativecommons.org/licenses/by/2.0/):
  * "Cutting back to one cup of coffee today ;-)" by Daniel Farrell: https://www.flickr.com/photos/danielfarrell/4387451834
  * "Fish Dish" by Yosomono: https://www.flickr.com/photos/preetamrai/236026381
  * "Sunbound" by Kyle Garrity: https://www.flickr.com/photos/kylegarrity/8218945454
  * "Tasman Peninsula" by Prince Roy: https://www.flickr.com/photos/princeroy/2074808383

  The images are unmodified apart from storage.
* **AI-generated images** come from the Community Forensics evaluation set
  (https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval), released under **CC BY-NC-SA 4.0**. They may be
  used for non-commercial research only. The terms of the original generator services may also apply.
