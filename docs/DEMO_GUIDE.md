# Demo guide (5 minutes)

A runbook for presenting the detector live, with what to say, what the audience will see, and what to do when
something goes wrong. All numbers quoted here are measured; see [MODEL_AND_EVALUATION.md](MODEL_AND_EVALUATION.md).

## Before the demo

**The day before**
- [ ] `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1` has completed on the demo laptop.
- [ ] `.venv\Scripts\python scripts\check_env.py --smoke` shows only `[OK  ]` lines.
- [ ] Run through this script once, end to end.
- [ ] Copy one or two of your own photos (from a phone) to the desktop as optional live uploads.

**15 minutes before**
- [ ] Plug in the charger (GPU laptops throttle on battery). Close other heavy apps.
- [ ] `powershell -ExecutionPolicy Bypass -File scripts\run.ps1` and wait for the browser tab.
- [ ] Click **Sheep on a hillside** once, so the model is loaded and warm, then click **↺ New image**.
- [ ] Set the browser zoom to 100-110 % and make the window full screen (F11).
- [ ] Keep `docs/screenshots/` open in a second window as a backup.

No internet connection is needed once setup is done.

## The 5-minute script

| time | do | say |
|---|---|---|
| 0:00-0:30 | Show the home screen | "This estimates whether a photo-like image came from a camera or an AI generator. It gives a calibrated probability, abstains when unsure, shows where the model looked, and tells you how far to trust it. It runs entirely on this laptop." |
| 0:30-1:30 | Click **Sheep on a hillside** | "The verdict is *Likely AI-generated*, 86 %. The dashed box is the inconclusive band: between 19 % and 73 % we refuse to decide. On our held-out benchmark, images scored 80-90 % were AI 97 % of the time." Scroll down: "The dataset label confirms it: Stable Cascade - a generator family the model never saw in training." |
| 1:30-2:15 | **Explanation** tab | "Grad-CAM shows which regions pushed the score towards AI - here the sheep texture. Dimmed areas were not examined. This shows where *the model* looked; it is not proof of editing. We measured that hiding these regions changes the score more than hiding random ones in 76 % of cases." |
| 2:15-2:45 | **↺ New image**, click **Coffee cup** | "A real Flickr photo from Open Images: 4 % AI, *Likely real*." |
| 2:45-3:30 | **↺ New image**, click **City street (fisheye)** | "Now a failure, on purpose. This is an Imagen 3 image, but the model says 11 % AI. Imagen 3 and Midjourney are our weakest generators: we catch only about half. We show this rather than hide it." |
| 3:30-4:30 | **Reliability** tab | "These are the measured numbers on 3,784 held-out images: ROC-AUC 0.909, 72.5 % of AI images caught, 88.8 % of real photos kept real. When the app does decide, it is right 93 % of the time, but it abstains on 45 % of images. The bars show which generators and photo sources are weak." |
| 4:30-5:00 | Close | "Limits: weak on Imagen and Midjourney, low-resolution images, and some older web photos. Mid-range scores understate AI. It is a research prototype - an aid, not proof." |

**If there is extra time:** upload your own phone photo (expect *likely real* or *inconclusive* - say so honestly
whichever it is), show **Analysis breakdown** (per-region scores), or show the CLI:
`.venv\Scripts\python predict.py samples\ai_flux_dev_pattern.png`.

## What each demo image shows

| button | true label (from the dataset) | app result | use it to show |
|---|---|---|---|
| Sheep on a hillside | AI (Stable Cascade) | 86 % - likely AI | a correct AI detection; partial-coverage warning on a large image |
| Coffee cup | real (Open Images, Flickr) | 4 % - likely real | a correct real photo |
| Floral pattern | AI (FLUX-dev) | 94 % - likely AI | a non-photographic AI image |
| Fish dish | real (Open Images, Flickr) | 0.4 % - likely real | a confident real photo |
| Potted plant | AI (Midjourney v6.1) | 45 % - inconclusive | the abstain band; Midjourney is hard |
| Two animals | real (Open Images, Flickr) | 21 % - inconclusive | the band also protects real photos |
| Sunflower and bee | real (Open Images, Flickr) | 82 % - likely AI (**wrong**) | a false alarm |
| City street (fisheye) | AI (Imagen 3) | 11 % - likely real (**wrong**) | a missed AI image |

Scores were produced by the current model on the development laptop's GPU. On the CPU the same images score within
half a percentage point (largest difference: Potted plant, 44.9 % on the GPU, 45.3 % on the CPU); no verdict changes.

## Fallbacks

| problem during the demo | do this |
|---|---|
| CUDA / GPU error message in the app | stop with Ctrl+C, `scripts\run.ps1 -Cpu`. Analysis takes about 0.4 s and the heat map about 2 s on the CPU |
| `port 8501 is already in use` | `scripts\run.ps1 -Port 8502` (an old copy is probably still running) |
| Browser tab did not open | open <http://localhost:8501> yourself |
| App is very slow on the first click | that is the model loading once; the next clicks are fast |
| App will not start at all | use the CLI: `.venv\Scripts\python predict.py samples\ai_stable_cascade_sheep.png` - same model, same numbers |
| Laptop fails completely | present the real screenshots in `docs/screenshots/` and the tables in the README |
| Your own upload is refused | only JPG, PNG, WEBP, BMP and TIFF up to 25 MB are accepted (on purpose); use a demo image |
| An uploaded photo is called AI | say so: real photos are wrongly flagged about 11 % of the time at the 50 % threshold (3 % with the band). Show the per-region scores and the Reliability tab |

## Questions you are likely to get

See [FAQ_AND_VIVA.md](FAQ_AND_VIVA.md). The short answers:
* **"How accurate is it?"** ROC-AUC 0.909 on generators it never saw; 72.5 % of AI images caught and 88.8 % of real photos
  kept real at the 50 % threshold; 93 % right when it decides, abstaining on 45 %.
* **"Can it detect deepfake videos / face swaps?"** No. It was built and measured for whole AI-generated images.
* **"Is the heat map where the image was faked?"** No - it shows what influenced the model.
* **"Why trust the probability?"** It is temperature-calibrated on a separate split; the Reliability tab shows the
  calibration curve, including where it is off.
