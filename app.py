"""AI Image Detector - Streamlit UI.

    streamlit run app.py

Everything numeric on the page comes from detector.analyze()/detector.explain() (shared with predict.py) or from the
measured model card artifacts/model_card_<model>.json written by make_model_card.py. Nothing about accuracy is hard-coded.
Rollback to the previous UI/engine: DETECTOR_BACKEND=legacy (needs TensorFlow) - see legacy/app_legacy.py.
"""
import glob
import hashlib
import io
import json
import os
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageChops, ImageDraw, ImageEnhance

if os.environ.get("DETECTOR_BACKEND", "").lower() == "legacy":
    import runpy
    runpy.run_path(os.path.join("legacy", "app_legacy.py"))
    st.stop()

import detector  # noqa: E402

st.set_page_config(page_title="AI Image Detector", page_icon="🔍", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1180px;}
  .hero-title {font-size: 2rem; font-weight: 750; letter-spacing: -0.02em; margin: 0;}
  .hero-sub {opacity: .72; margin: .25rem 0 1.2rem 0; font-size: 1rem;}
  .card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 1.1rem 1.25rem; background: rgba(128,128,128,.06);}
  .chip {display: inline-block; padding: .35rem .9rem; border-radius: 999px; font-weight: 700; font-size: 1.15rem; border: 1.5px solid;}
  .chip.fake {color: #dc2626; background: rgba(220,38,38,.12);}
  .chip.real {color: #16a34a; background: rgba(22,163,74,.12);}
  .chip.unsure {color: #d97706; background: rgba(217,119,6,.14);}
  .score {font-size: 2.6rem; font-weight: 750; line-height: 1.1; margin: .5rem 0 0 0;}
  .score small {font-size: .95rem; font-weight: 500; opacity: .7;}
  .gauge {position: relative; height: 14px; border-radius: 7px; margin: .9rem 0 .3rem 0;
          background: linear-gradient(90deg, #16a34a 0%, #84cc16 30%, #f59e0b 50%, #f97316 70%, #dc2626 100%);}
  .gauge .band {position: absolute; top: -3px; bottom: -3px; border: 2px dashed rgba(128,128,128,.9); border-radius: 4px;}
  .gauge .mark {position: absolute; top: -6px; width: 4px; height: 26px; border-radius: 2px; background: currentColor;
                box-shadow: 0 0 0 2px rgba(255,255,255,.7);}
  .gauge-lab {display: flex; justify-content: space-between; font-size: .78rem; opacity: .7;}
  .tag {font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; opacity: .6; margin-bottom: .2rem;}
</style>
""", unsafe_allow_html=True)

MODELS_DIR = "models"
SAMPLE_DIR = "test_images"


# ----------------------------------------------------------------------------- helpers
def list_models():
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(MODELS_DIR, "*.pt")))
    names.sort(key=lambda n: (not n.startswith("general_"), n))
    return names


def model_card(model_file):
    path = os.path.join("artifacts", f"model_card_{os.path.splitext(model_file)[0]}.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"


@st.cache_data(show_spinner=False, max_entries=6)
def run_analysis(data: bytes, model_file: str):
    im = Image.open(io.BytesIO(data))
    info = dict(format=im.format, mode=im.mode, size=im.size, bytes=len(data),
                exif_fields=len(im.getexif()) if hasattr(im, "getexif") else 0, jpeg_q=detector.jpeg_quality(im))
    t0 = time.time()
    res = detector.analyze(im, model=model_file)
    t1 = time.time()
    exp = detector.explain(Image.open(io.BytesIO(data)), model=model_file,
                           evidence_for="real" if res.real_prob >= 0.5 else "fake")
    return res, exp, info, dict(analyze_s=t1 - t0, explain_s=time.time() - t1)


def fft_view(pil):
    import cv2
    g = np.asarray(pil.convert("L").resize((256, 256)))
    mag = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))) + 1e-8)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def ela_view(pil, quality=90):
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=quality)
    diff = ImageChops.difference(pil.convert("RGB"), Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    top = max(e[1] for e in diff.getextrema())
    return ImageEnhance.Brightness(diff).enhance(255.0 / max(top, 1))


def gauge_html(p_fake, band):
    left = f"{100 * p_fake:.1f}%"
    shade = ""
    if band:
        lo_f, hi_f = 1 - band[1], 1 - band[0]          # band is on P(real); the gauge runs along P(AI)
        shade = f'<div class="band" style="left:{100 * lo_f:.1f}%;width:{100 * (hi_f - lo_f):.1f}%"></div>'
    return (f'<div class="gauge">{shade}<div class="mark" style="left:calc({left} - 2px)"></div></div>'
            '<div class="gauge-lab"><span>0% - looks camera-made</span><span>looks AI-generated - 100%</span></div>')


def tiles_figure(exp, res):
    """Prepared image with the analysed 256 px regions outlined and their P(AI)."""
    base = Image.fromarray(exp.base)
    s = base.size[0] / res.prepared_size[0]
    d = ImageDraw.Draw(base)
    for (x0, y0, x1, y1), p in zip(res.boxes, res.tile_real):
        col = (220, 38, 38) if p < 0.5 else (22, 163, 74)
        d.rectangle([x0 * s, y0 * s, x1 * s - 1, y1 * s - 1], outline=col, width=max(2, round(base.size[0] / 300)))
        d.text((x0 * s + 6, y0 * s + 4), f"AI {100 * (1 - p):.0f}%", fill=col)
    return base


# ----------------------------------------------------------------------------- sidebar
models = list_models()
active = detector.active_model_name()
with st.sidebar:
    st.header("Settings")
    if not models:
        st.error("No checkpoint found in models/.")
        st.stop()
    model_file = st.selectbox("Detector", models, index=models.index(active) if active in models else 0,
                              help="general_* models analyse any kind of image. The others are the original "
                                   "face-only models, kept for comparison and rollback.")
    st.caption("Default is models/active_model.json. Same model & settings are used by `python predict.py`.")
    st.divider()
    st.markdown("**How to read the result**")
    st.markdown("- *Prediction*: the verdict, from the calibrated score.\n"
                "- *Visual evidence*: where the **model** looked.\n"
                "- *Forensic observations*: generic checks the model does **not** use.\n"
                "- *Reliability*: measured accuracy and known failures.")

card = model_card(model_file)

# ----------------------------------------------------------------------------- header + input
st.markdown('<p class="hero-title">🔍 AI Image Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Estimates whether an image was made by a camera or generated by AI, shows where the '
            'model looked, and says how far to trust the answer. A research prototype - not forensic proof.</p>',
            unsafe_allow_html=True)

up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
                      help="Everything is processed locally; nothing is uploaded anywhere.")
samples = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.*")))
if samples:
    with st.expander("Or try a bundled image (ground truth not shown on purpose)"):
        cols = st.columns(min(4, len(samples)))
        for i, p in enumerate(samples):
            if cols[i % len(cols)].button(os.path.basename(p), key=f"sample_{i}", width="stretch"):
                st.session_state["sample"] = p

data, name = None, None
if up is not None:
    data, name = up.getvalue(), up.name
    st.session_state.pop("sample", None)
elif st.session_state.get("sample") and os.path.exists(st.session_state["sample"]):
    with open(st.session_state["sample"], "rb") as f:
        data, name = f.read(), os.path.basename(st.session_state["sample"])

if data is None:
    st.info("Upload an image or pick a bundled one to start.")
    st.stop()

try:
    with st.spinner("Analysing - scoring regions and building the attention map..."):
        res, exp, info, timing = run_analysis(data, model_file)
except Exception as e:                                              # unreadable / unsupported file
    st.error(f"Could not analyse this file: {e}")
    st.stop()

p_fake = res.fake_prob
face_kind = res.kind == "face"

# ----------------------------------------------------------------------------- 1. prediction
left, right = st.columns([5, 6], gap="large")
with left:
    st.image(detector.to_rgb(Image.open(io.BytesIO(data))), caption=f"{name} - {info['size'][0]}x{info['size'][1]} px",
             width="stretch")

with right:
    if face_kind:
        st.warning("This is the original **face-only** model. It was trained on face photographs and its answer is not "
                   "meaningful for other kinds of images. Select a `general_*` detector in the sidebar for general use.")
    verdict_cls = {"likely_fake": "fake", "likely_real": "real", "inconclusive": "unsure", "": "unsure"}[res.verdict]
    verdict_txt = {"likely_fake": "Likely AI-generated", "likely_real": "Likely real", "inconclusive": "Inconclusive",
                   "": ("Leans AI-generated" if p_fake >= 0.5 else "Leans real")}[res.verdict]
    st.markdown('<div class="tag">Prediction</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><span class="chip {verdict_cls}">{verdict_txt}</span>'
                f'<div class="score">{pct(p_fake)} <small>calibrated probability that this image is AI-generated</small></div>'
                f'{gauge_html(p_fake, res.band)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="tag" style="margin-top:1rem">Uncertainty</div>', unsafe_allow_html=True)
    if res.band:
        st.markdown(f"Scores between **{pct(1 - res.band[1])}** and **{pct(1 - res.band[0])}** AI-probability (dashed box) "
                    "are reported as *inconclusive*; the model is not allowed to guess there.")
    bins = (card or {}).get("reliability_bins") or []
    hit = next((b for b in bins if b["lo"] <= p_fake <= b["hi"] and b["n"] >= 20), None)
    if hit:
        st.markdown(f"On the held-out benchmark, images that received a score in this range ({pct(hit['lo'])}-{pct(hit['hi'])}) "
                    f"were actually AI-generated **{pct(hit['fake_rate'])}** of the time (n={hit['n']}).")
    elif card is None:
        st.markdown("No measured benchmark is attached to this model, so no reliability statement can be made.")
    if res.tile_real and len(res.tile_real) > 1:
        lo_t, hi_t = 1 - max(res.tile_real), 1 - min(res.tile_real)
        st.markdown(f"{len(res.tile_real)} image regions were scored separately; their AI-probabilities range from "
                    f"**{pct(lo_t)}** to **{pct(hi_t)}**. The headline is the average of their logits.")
    for code, text in res.flags:
        if code != "face_only_model":
            st.warning(text)

st.write("")
tab_vis, tab_for, tab_rel, tab_tech = st.tabs(["Visual evidence", "Forensic observations", "How reliable is this?",
                                               "Technical details"])

# ----------------------------------------------------------------------------- 2. visual evidence
with tab_vis:
    view = st.radio("View", ["Model attention", "Original", "Regions analysed"], horizontal=True,
                    label_visibility="collapsed")
    if view == "Model attention":
        st.image(exp.overlay, width="stretch",
                 caption=f"Grad-CAM of the model's own output for '{exp.evidence_for}' (layer {exp.layer}). "
                         "Warm colours = regions that pushed the model toward that answer. Dimmed = not examined.")
    elif view == "Original":
        st.image(exp.base, width="stretch")
    elif res.boxes:
        st.image(tiles_figure(exp, res), width="stretch",
                 caption="The 256 px regions the network actually saw (full resolution, no resizing) and each one's score.")
    else:
        st.info("This model looks at the whole image at once, so there are no separate regions.")
    st.markdown("**What this map is - and is not.** It shows which parts of the image influenced *this model's* score. It "
                "is **not** proof that those pixels were edited or generated, and it can highlight perfectly ordinary areas.")
    ex = (card or {}).get("explain")
    if ex:
        st.markdown(f"*Measured connection to the decision:* on {ex['n']} validation images, removing the highlighted "
                    f"regions lowered the model's own score more than removing random regions in "
                    f"**{pct(ex['beats_random_frac'])}** of cases (chance level is about 25%). This verifies the map "
                    "tracks the model, not that it tracks manipulation.")
    if res.tile_real and len(res.tile_real) > 1:
        with st.expander("Per-region scores"):
            rows = [dict(region=i + 1, x=b[0], y=b[1], ai_probability=round(100 * (1 - p), 1))
                    for i, (b, p) in enumerate(zip(res.boxes, res.tile_real))]
            st.dataframe(rows, width="stretch", hide_index=True)

# ----------------------------------------------------------------------------- 3. forensic observations
with tab_for:
    st.markdown("These are **independent, generic checks of the file**. The detector does **not** use them, none has a "
                "validated threshold, and ordinary resizing or re-compression can produce the same patterns - so they "
                "cannot confirm or refute the prediction.")
    f1, f2 = st.columns(2)
    f1.metric("File format", info["format"] or "unknown")
    f2.metric("Size", f"{info['size'][0]}x{info['size'][1]} px / {info['bytes'] / 1024:.0f} KB")
    f3, f4 = st.columns(2)
    f3.metric("Estimated JPEG quality", "n/a (not a JPEG)" if info["jpeg_q"] is None else f"~{info['jpeg_q']:.0f}")
    f4.metric("EXIF fields", str(info["exif_fields"]))
    st.caption("Missing EXIF is common for both real photos (stripped by messengers, social media) and AI images, "
               "so its absence says nothing; present camera EXIF can be edited. Neither is used by the detector.")
    pil_full = detector.to_rgb(Image.open(io.BytesIO(data)))
    with st.expander("Error Level Analysis (generic diagnostic)"):
        st.image(ela_view(pil_full), width="stretch",
                 caption="Difference after re-saving at JPEG quality 90, contrast-stretched. Reflects the compression history "
                         "and image content (edges, texture). It is not evidence of AI generation.")
    with st.expander("Frequency spectrum (generic diagnostic)"):
        st.image(fft_view(pil_full), width=320,
                 caption="Log power spectrum of the grayscale image resized to 256 px. Resizing and compression change it; "
                         "no decision rule is attached to it here.")

# ----------------------------------------------------------------------------- 4. reliability
with tab_rel:
    if card is None:
        st.warning("No measured model card is available for this model (artifacts/model_card_*.json). "
                   "Treat its output as unvalidated.")
    else:
        b = card["benchmark"]
        st.markdown(f"**Independent benchmark:** {b['name']} - {b['n_fake']} AI images from generator families the model "
                    f"never saw in training + {b['n_real']} real images. {card.get('benchmark_note', '')}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{b['auc']:.2f}", help=f"95% CI {b['auc_ci'][0]:.2f}-{b['auc_ci'][1]:.2f}")
        m2.metric("AI images caught", pct(b["fake_recall"]))
        m3.metric("Real images kept real", pct(b["real_specificity"]))
        m4.metric("Calibration error (ECE)", f"{b['ece']:.2f}")
        st.caption(f"95% CI for AUC: {b['auc_ci'][0]:.2f}-{b['auc_ci'][1]:.2f}. Percentages use the default decision "
                   "(P(AI) = 50%).")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Detection rate by AI-generator family**")
            st.dataframe([{"family": k, "images": v["n"], "caught": pct(v["recall"])}
                          for k, v in sorted(card["per_family"].items(), key=lambda kv: kv[1]["recall"])],
                         hide_index=True, width="stretch")
        with c2:
            st.markdown("**Real images kept real, by source**")
            st.dataframe([{"source": k, "images": v["n"], "kept real": pct(v["specificity"])}
                          for k, v in sorted(card["per_real_source"].items(), key=lambda kv: kv[1]["specificity"])],
                         hide_index=True, width="stretch")
        st.markdown("**Robustness (same images, degraded)**")
        st.dataframe([{"condition": k, "ROC-AUC": round(v["auc"], 3), "balanced accuracy": pct(v["balanced_acc"])}
                      for k, v in card["robustness"].items()], hide_index=True, width="stretch")
        weak = [k for k, v in card["per_family"].items() if v["recall"] < 0.5]
        if weak:
            st.error("Documented weak spots (detection rate below 50%): " + ", ".join(sorted(weak)) + ".")
        st.markdown("**Known limits**")
        for line in card.get("limitations", []):
            st.markdown(f"- {line}")

# ----------------------------------------------------------------------------- 5. technical
with tab_tech:
    st.markdown(f"- **Model**: `{res.model_id}` ({res.kind})")
    st.markdown(f"- **Temperature (calibration)**: {res.temperature:.3f} - fitted on a held-out calibration split, "
                "not on the benchmark")
    if res.band:
        st.markdown(f"- **Decision band (P(real))**: inconclusive between {res.band[0]:.2f} and {res.band[1]:.2f}")
    if res.tile_real:
        st.markdown(f"- **Regions scored**: {len(res.tile_real)} of 256x256 px on a {res.prepared_size[0]}x{res.prepared_size[1]} "
                    "px working image (flip-averaged)")
    st.markdown(f"- **Run time**: analysis {timing['analyze_s']:.2f}s, attention map {timing['explain_s']:.2f}s")
    st.markdown(f"- **File SHA-256**: `{hashlib.sha256(data).hexdigest()[:16]}...`")
    st.caption("The command-line tool `python predict.py <image>` runs the same code path and gives the same numbers.")

st.divider()
st.caption("Research prototype. Results are probabilistic, can be wrong, and must not be used as sole evidence about "
           "anyone's image. Training and benchmark data are licensed for non-commercial research (see README).")
