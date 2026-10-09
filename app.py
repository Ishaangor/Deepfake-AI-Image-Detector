"""AI Image Detector - Streamlit web interface.

    streamlit run app.py            (or scripts\\run.ps1)

Every number on the page comes from detector.analyze() / detector.explain() (the same code path as predict.py) or from
the measured model card models/<model>.card.json written by evaluation/model_card.py. No accuracy figure is hard-coded.
Rollback to the original UI/engine: set DETECTOR_BACKEND=legacy (needs TensorFlow) - see legacy/app_legacy.py.
"""
import hashlib
import json
import os
import sys
import time

import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if os.environ.get("DETECTOR_BACKEND", "").lower() == "legacy":
    import runpy
    runpy.run_path(os.path.join(ROOT, "legacy", "app_legacy.py"))
    st.stop()

import detector  # noqa: E402
from detector import forensics  # noqa: E402
from detector.paths import SAMPLES  # noqa: E402

st.set_page_config(page_title="AI Image Detector", page_icon="🔍", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 3.6rem; padding-bottom: 2rem; max-width: 1240px;}
  .hero {display: flex; align-items: center; gap: .9rem; margin-bottom: .2rem;}
  .hero .logo {font-size: 2.1rem; line-height: 1;}
  .hero h1 {font-size: 1.85rem; font-weight: 750; letter-spacing: -0.02em; margin: 0; padding: 0;}
  .hero-sub {opacity: .75; margin: .15rem 0 .7rem 0; font-size: 1rem; max-width: 60rem;}
  .pills {display: flex; gap: .45rem; flex-wrap: wrap; margin: 0 0 1rem 0;}
  .pill {font-size: .8rem; padding: .2rem .65rem; border-radius: 999px; border: 1px solid rgba(128,128,128,.35);
         background: rgba(128,128,128,.08);}
  .tag {font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; opacity: .62; margin: 0 0 .25rem 0;
        font-weight: 600;}
  .verdict {display: inline-flex; align-items: center; gap: .5rem; padding: .45rem 1rem; border-radius: 12px;
            font-weight: 750; font-size: 1.35rem; border: 1.5px solid;}
  .verdict.fake {color: #dc2626; background: rgba(220,38,38,.10);}
  .verdict.real {color: #16a34a; background: rgba(22,163,74,.10);}
  .verdict.unsure {color: #d97706; background: rgba(217,119,6,.12);}
  .score {font-size: 2.7rem; font-weight: 780; line-height: 1.05; margin: .7rem 0 0 0; font-variant-numeric: tabular-nums;}
  .score small {display: block; font-size: .95rem; font-weight: 500; opacity: .72; line-height: 1.35; margin-top: .2rem;}
  .gauge {position: relative; height: 14px; border-radius: 7px; margin: 1rem 0 .3rem 0;
          background: linear-gradient(90deg, #16a34a 0%, #84cc16 28%, #eab308 50%, #f97316 72%, #dc2626 100%);}
  .gauge .band {position: absolute; top: -4px; bottom: -4px; border: 2px dashed rgba(128,128,128,.95); border-radius: 5px;
                background: rgba(255,255,255,.18);}
  .gauge .mark {position: absolute; top: -7px; width: 5px; height: 28px; border-radius: 3px; background: #111827;
                box-shadow: 0 0 0 2px #ffffff;}
  .gauge-lab {display: flex; justify-content: space-between; gap: .8rem; font-size: .78rem; opacity: .72;}
  .gauge-lab span:nth-child(2) {text-align: center;}
  .gauge-lab span:last-child {text-align: right;}
  .truth {margin-top: .8rem; padding: .55rem .8rem; border-radius: 10px; border: 1px dashed rgba(128,128,128,.5);
          font-size: .92rem;}
  .step {font-size: .92rem;}
  .step b {display: block; font-size: 1rem; margin-bottom: .15rem;}
  div[data-testid="stFileUploaderDropzone"] {padding: 1.6rem 1rem;}
</style>
""", unsafe_allow_html=True)

VERDICTS = {  # verdict code -> (headline, css class, plain-language meaning)
    "likely_fake": ("Likely AI-generated", "fake",
                    "The calibrated score is above the upper edge of the inconclusive band."),
    "likely_real": ("Likely real (camera)", "real",
                    "The calibrated score is below the lower edge of the inconclusive band."),
    "inconclusive": ("Inconclusive", "unsure",
                     "The score falls inside the band where the model is not reliable enough to decide."),
}
TABS = ["Explanation", "Analysis breakdown", "File diagnostics", "Reliability", "Method & limits", "Technical"]


# ----------------------------------------------------------------------------- helpers
def pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"


@st.cache_data(show_spinner=False)
def load_samples():
    try:
        with open(os.path.join(SAMPLES, "manifest.json"), encoding="utf-8") as f:
            items = json.load(f)["samples"]
    except (OSError, ValueError, KeyError):
        return []
    return [s for s in items if os.path.isfile(os.path.join(SAMPLES, s["file"]))]


@st.cache_data(show_spinner=False)
def thumbnail(path, side=320):
    im = detector.load_rgb(path)
    im.thumbnail((side, side))
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


@st.cache_data(show_spinner=False, max_entries=8)
def run_analysis(data: bytes, model_file: str):
    info = forensics.file_info(data)
    t0 = time.time()
    res = detector.analyze(forensics.open_upload(data), model=model_file)
    return res, info, time.time() - t0


@st.cache_data(show_spinner=False, max_entries=16)
def run_explain(data: bytes, model_file: str, evidence_for: str):
    t0 = time.time()
    exp = detector.explain(forensics.open_upload(data), model=model_file, evidence_for=evidence_for)
    return exp, time.time() - t0


def gauge_html(p_fake, band):
    shade = ""
    if band:
        lo_f, hi_f = 1 - band[1], 1 - band[0]                     # band is stored on P(real); the gauge runs along P(AI)
        shade = f'<div class="band" style="left:{100 * lo_f:.1f}%;width:{100 * (hi_f - lo_f):.1f}%"></div>'
    return (f'<div class="gauge">{shade}<div class="mark" style="left:calc({100 * p_fake:.1f}% - 2px)"></div></div>'
            '<div class="gauge-lab"><span>0% &middot; camera</span><span>dashed = inconclusive</span>'
            '<span>AI &middot; 100%</span></div>')


def regions_figure(base, res):
    """Analysed image with every 256 px region outlined and labelled with its own AI-probability."""
    img = Image.fromarray(base)
    s = img.size[0] / res.prepared_size[0]
    d = ImageDraw.Draw(img)
    for (x0, y0, x1, y1), p in zip(res.boxes, res.tile_real):
        col = (220, 38, 38) if p < 0.5 else (22, 163, 74)
        d.rectangle([x0 * s, y0 * s, x1 * s - 1, y1 * s - 1], outline=col, width=max(2, round(img.size[0] / 300)))
        d.rectangle([x0 * s, y0 * s, x0 * s + 64, y0 * s + 16], fill=(255, 255, 255))
        d.text((x0 * s + 4, y0 * s + 3), f"AI {100 * (1 - p):.0f}%", fill=col)
    return img


def checks(res, info):
    """Input-quality checklist. Uses exactly the flags detector.analyze() raised, plus the passes for the same checks."""
    codes = {c for c, _ in res.flags}
    text = dict(res.flags)
    w, h = info["size"]
    out = [("Resolution", "low_res" not in codes, text.get("low_res", f"{w}x{h} px - at least 256 px on the short side."))]
    q = info["jpeg_q"]
    out.append(("Compression", "heavy_jpeg" not in codes,
                text.get("heavy_jpeg", "Not a JPEG." if q is None else f"JPEG quality about {q:.0f} - not heavily compressed.")))
    out.append(("Colour", "grayscale" not in codes, text.get("grayscale", "Colour image.")))
    if res.kind == "general":
        out.append(("Coverage", "partial_coverage" not in codes,
                    text.get("partial_coverage", "Most of the image was examined.")))
        out.append(("Region agreement", "tiles_disagree" not in codes,
                    text.get("tiles_disagree", "The analysed regions do not point in opposite directions.")))
    else:
        out.append(("Model applicability", False, text.get("face_only_model", "Face-only model.")))
    return out


def reset():
    st.session_state.pop("sample", None)
    st.session_state["upload_key"] = st.session_state.get("upload_key", 0) + 1


# ----------------------------------------------------------------------------- sidebar
models = detector.list_models()
active = detector.active_model_name()
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    if not models:
        st.error("No model checkpoint found in `models/`. Run `scripts\\setup.ps1` (it verifies the model files) or "
                 "see `models/README.md`.")
        st.stop()
    model_file = st.selectbox(
        "Detector model", models, index=models.index(active) if active in models else 0,
        format_func=lambda n: f"{n}  (default)" if n == active else n,
        help="general_* models analyse any kind of image. The other checkpoints are the project's earlier face-only "
             "models, kept for comparison and rollback.")
    st.caption(f"Running on **{detector.device().upper()}**. The CLI (`python predict.py`) uses the same model and code.")
    st.divider()
    st.markdown("### 📖 How to read a result")
    st.markdown("1. **Verdict** - from a *calibrated* probability and an *inconclusive* band.\n"
                "2. **Explanation** - where the *model* looked (not proof of editing).\n"
                "3. **Breakdown** - input problems that make the answer less reliable.\n"
                "4. **Reliability** - measured accuracy and known weak spots.")
    st.divider()
    st.caption("Research prototype for non-commercial use. Images are processed locally and are not uploaded anywhere.")

card = detector.load_model_card(model_file)

# ----------------------------------------------------------------------------- header
st.markdown('<div class="hero"><span class="logo">🔍</span><h1>AI Image Detector</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Estimates whether a photo-like image was <b>captured by a camera</b> or <b>generated by an '
            'AI model</b>, shows which regions influenced the decision, and tells you how far the answer can be trusted.</p>',
            unsafe_allow_html=True)
bench = (card or {}).get("benchmark")
pills = ["Calibrated probability", "Abstains when unsure", "Grad-CAM explanation", "Runs locally"]
if bench:
    pills.insert(0, f"ROC-AUC {bench['auc']:.3f} on held-out generators")
st.markdown('<div class="pills">' + "".join(f'<span class="pill">{p}</span>' for p in pills) + "</div>",
            unsafe_allow_html=True)

# ----------------------------------------------------------------------------- input
samples = load_samples()
data, name, sample = None, None, None
uploaded = None
if not st.session_state.get("sample"):
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
        key=f"upload_{st.session_state.get('upload_key', 0)}",
        help="Drag and drop or browse. JPG, PNG, WEBP, BMP or TIFF up to 25 MB. Best results: an original file at "
             "256 px or more on the short side, not a screenshot or a heavily compressed re-upload.")
if uploaded is not None:
    data, name = uploaded.getvalue(), uploaded.name
elif st.session_state.get("sample"):
    sample = next((s for s in samples if s["file"] == st.session_state["sample"]), None)
    if sample:
        with open(os.path.join(SAMPLES, sample["file"]), "rb") as f:
            data, name = f.read(), sample["display_name"]

if data is None:
    if samples:
        st.markdown('<p class="tag">Or try a demo image</p>', unsafe_allow_html=True)
        st.caption("Labelled images from held-out data (never used for training). The true label is revealed after the "
                   "analysis. The set deliberately includes images the model gets wrong.")
        cols = st.columns(4)
        for i, s in enumerate(samples):
            with cols[i % 4]:
                st.image(thumbnail(os.path.join(SAMPLES, s["file"])), width="stretch")
                if st.button(s["display_name"], key=f"sample_{s['file']}", width="stretch"):
                    st.session_state["sample"] = s["file"]
                    st.rerun()
    st.write("")
    a, b, c = st.columns(3)
    a.markdown('<div class="step"><b>1 · Upload</b>Drop a photo-like image. It is read locally, never sent anywhere.</div>',
               unsafe_allow_html=True)
    b.markdown('<div class="step"><b>2 · Analyse</b>Up to nine 256 px regions are scored at native resolution and '
               'averaged into one calibrated probability.</div>', unsafe_allow_html=True)
    c.markdown('<div class="step"><b>3 · Interpret</b>Read the verdict with its uncertainty, the attention map and the '
               'measured limits of the model.</div>', unsafe_allow_html=True)
    st.stop()

top_l, top_r = st.columns([4, 1])
top_l.markdown(f'<p class="tag">Analysing</p><b>{name}</b>', unsafe_allow_html=True)
top_r.button("↺ New image", help="Clear this result and analyse another image", on_click=reset, width="stretch", type="primary", key="reset")

try:
    with st.spinner("Scoring image regions..."):
        res, info, t_analyze = run_analysis(data, model_file)
except UnidentifiedImageError:
    st.error("This file could not be read as an image. Please upload a JPG, PNG, WEBP, BMP or TIFF file.")
    st.stop()
except FileNotFoundError as e:
    st.error(f"{e}. Run `scripts\\setup.ps1` to verify the model files.")
    st.stop()
except Exception as e:                                            # corrupt file, out of memory, GPU problems ...
    st.error(f"Analysis failed: {e}. If this mentions CUDA or memory, restart with `scripts\\run.ps1 -Cpu`.")
    st.stop()

p_fake = res.fake_prob
headline, css, meaning = VERDICTS.get(res.verdict, ("Leans AI-generated" if p_fake >= 0.5 else "Leans real", "unsure",
                                                    "This model has no inconclusive band; the 50% threshold is used."))

# ----------------------------------------------------------------------------- 1. result
left, right = st.columns([5, 6], gap="large")
with left:
    st.image(detector.to_rgb(forensics.open_upload(data)), width="stretch",
             caption=f"{info['size'][0]}x{info['size'][1]} px · {info['format'] or 'unknown format'} · "
                     f"{info['bytes'] / 1024:.0f} KB")

with right:
    if res.kind == "face":
        st.warning("This is one of the original **face-only** models. It was trained on face photographs only, so its "
                   "answer is not meaningful for other images. Choose a `general_*` model in the sidebar.")
    st.markdown('<p class="tag">Prediction</p>', unsafe_allow_html=True)
    st.markdown(f'<span class="verdict {css}">{headline}</span>'
                f'<div class="score">{pct(p_fake)}<small>calibrated probability that this image is AI-generated</small></div>'
                f'{gauge_html(p_fake, res.band)}', unsafe_allow_html=True)
    st.write("")
    st.markdown('<p class="tag">What this means</p>', unsafe_allow_html=True)
    lines = [meaning]
    if res.band:
        lines.append(f"Scores between **{pct(1 - res.band[1])}** and **{pct(1 - res.band[0])}** are reported as "
                     "*inconclusive* (dashed box). The band was set on a separate calibration set so that decided "
                     "answers there were right about 90% of the time.")
    bins = (card or {}).get("reliability_bins") or []
    hit = next((b for b in bins if b["lo"] <= p_fake <= b["hi"] and b["n"] >= 20), None)
    if hit:
        lines.append(f"On the held-out benchmark, images scored {pct(hit['lo'])}-{pct(hit['hi'])} were actually "
                     f"AI-generated **{pct(hit['fake_rate'])}** of the time (n = {hit['n']}).")
    lines.append("A probability is an estimate, not proof. It can be confidently wrong - see *Reliability*.")
    st.markdown("\n".join(f"- {x}" for x in lines))
    for code, text in res.flags:
        if code != "face_only_model":
            st.warning(text, icon="⚠️")
    if sample:
        truth = "AI-generated" if sample["label"] == "ai_generated" else "Real photograph"
        gen = f" ({sample['generator']})" if sample.get("generator") else ""
        st.markdown(f'<div class="truth"><b>Dataset label:</b> {truth}{gen}<br><span style="opacity:.75">Source: '
                    f'{sample["label_source"]}. Licence: {sample["license"]}.</span></div>', unsafe_allow_html=True)

st.write("")
tab_exp, tab_break, tab_diag, tab_rel, tab_method, tab_tech = st.tabs(TABS)

# ----------------------------------------------------------------------------- 2. explanation
with tab_exp:
    c1, c2, c3 = st.columns([2, 2, 3])
    target = c1.segmented_control("Explain evidence for", ["AI-generated", "Real"],
                                  default="AI-generated" if p_fake >= 0.5 else "Real", key="target")
    view = c2.segmented_control("View", ["Heat map", "Regions", "Original"], default="Heat map", key="view")
    opacity = c3.slider("Heat-map opacity", 0.0, 1.0, 0.6, 0.05, key="opacity")
    target = target or "AI-generated"
    evidence = "fake" if target == "AI-generated" else "real"
    with st.spinner("Computing the attention map..."):
        exp, t_explain = run_explain(data, model_file, evidence)
    pic, txt = st.columns([3, 2], gap="large")
    if (view or "Heat map") == "Heat map":
        pic.image(forensics.blend_heatmap(exp.base, exp.heat, exp.examined, opacity), width="stretch",
                  caption=f"Grad-CAM for '{target}' (layer {exp.layer}). Coloured = regions that pushed the model "
                          "toward that answer (red strongest). Dimmed = not examined by the model.")
    elif view == "Regions" and res.boxes:
        pic.image(regions_figure(exp.base, res), width="stretch",
                  caption="The 256 px regions the network scored (native resolution, no resizing) and each region's "
                          "own AI-probability.")
    elif view == "Regions":
        pic.info("This model looks at the whole image at once, so there are no separate regions.")
    else:
        pic.image(exp.base, width="stretch")
    txt.markdown("**What the heat map is - and is not.** It shows which parts of the image influenced *this model's* "
                 "score. It is **not** proof that those pixels were edited or generated, and it can highlight perfectly "
                 "ordinary areas such as edges, text or texture.")
    txt.markdown("**How to use it.** Switch *Explain evidence for* to see what pushed the score the other way. "
                 "*Regions* shows each 256 px crop's own score, so you can see whether one area drives the verdict.")
    ex = (card or {}).get("explain")
    if ex:
        txt.caption(f"Measured link to the decision: on {ex['n']} validation images, hiding the highlighted regions "
                    f"lowered the model's own score more than hiding random regions in {pct(ex['beats_random_frac'])} "
                    "of cases. This checks that the map tracks the model - not that it tracks manipulation.")

# ----------------------------------------------------------------------------- 3. breakdown
with tab_break:
    st.markdown("**Input checks** - conditions under which the model is known to be less reliable.")
    for label, ok, text in checks(res, info):
        st.markdown(f"{'✅' if ok else '⚠️'} **{label}** - {text}")
    if res.tile_real and len(res.tile_real) > 1:
        st.markdown("**Per-region scores** - the headline is the average of the regions' log-odds.")
        rows = [dict(region=i + 1, x=b[0], y=b[1], ai_probability=round(100 * (1 - p), 1))
                for i, (b, p) in enumerate(zip(res.boxes, res.tile_real))]
        st.dataframe(rows, hide_index=True, width="stretch",
                     column_config={"ai_probability": st.column_config.ProgressColumn(
                         "AI probability (%)", min_value=0, max_value=100, format="%.1f")})
        lo_t, hi_t = 1 - max(res.tile_real), 1 - min(res.tile_real)
        st.caption(f"{len(res.tile_real)} regions, AI-probability from {pct(lo_t)} to {pct(hi_t)}. A wide spread means "
                   "different parts of the image look different to the model.")

# ----------------------------------------------------------------------------- 4. diagnostics
with tab_diag:
    st.info("These are **generic file checks the detector does not use**. None has a validated threshold here, and "
            "ordinary resizing, editing or re-compression produces the same patterns - so they cannot confirm or refute "
            "the prediction. They help describe the file.", icon="ℹ️")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Format", info["format"] or "unknown")
    m2.metric("Pixels", f"{info['size'][0]}x{info['size'][1]}")
    m3.metric("JPEG quality (est.)", "n/a" if info["jpeg_q"] is None else f"~{info['jpeg_q']:.0f}")
    m4.metric("EXIF fields", str(info["exif_fields"]))
    st.caption("EXIF is often stripped by messengers and social media, and AI images usually have none - so missing EXIF "
               "says nothing on its own, and present EXIF can be edited.")
    pil_full = detector.to_rgb(forensics.open_upload(data))
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Error Level Analysis (ELA)**")
        st.image(forensics.ela(pil_full), width="stretch",
                 caption="Difference after re-saving at JPEG quality 90, contrast-stretched. Bright = changed more. "
                         "Reflects compression history and content (edges, texture), not AI generation.")
    with d2:
        st.markdown("**Frequency spectrum**")
        st.image(forensics.spectrum(pil_full), width="stretch",
                 caption="Log power spectrum of the grayscale image at 256 px. Regular dot grids can come from periodic "
                         "patterns, resizing or JPEG blocks as well as from some generators; no rule is applied here.")

# ----------------------------------------------------------------------------- 5. reliability
with tab_rel:
    if card is None:
        st.warning("No measured model card is available for this model (models/<model>.card.json). "
                   "Treat its output as unvalidated.")
    else:
        import altair as alt
        import pandas as pd
        b = card["benchmark"]
        st.markdown(f"**Independent benchmark: {b['name']}** (project-constructed) - {b['n_fake']:,} AI-generated images "
                    f"+ {b['n_real']:,} real photographs. {card.get('benchmark_note', '')}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ROC-AUC", f"{b['auc']:.3f}", help=f"95% bootstrap CI {b['auc_ci'][0]:.3f}-{b['auc_ci'][1]:.3f}")
        k2.metric("AI images caught", pct(b["fake_recall"], 1), help="Recall on AI images at P(AI) >= 50%")
        k3.metric("Real images kept real", pct(b["real_specificity"], 1), help="Specificity at P(AI) >= 50%")
        k4.metric("Top-label ECE", f"{b['ece']:.3f}",
                  help="Expected calibration error of the decision's confidence (10 bins). See the class-wise figure below.")
        st.caption(f"ROC-AUC 95% CI {b['auc_ci'][0]:.3f}-{b['auc_ci'][1]:.3f}. Rates use the 50% threshold. A project-built "
                   "benchmark sample: it does not represent every generator or every kind of real photo.")
        if b.get("ece_classwise") is not None:
            st.warning(f"**Calibration caveat.** The top-label ECE ({b['ece']:.3f}) lets errors on both sides of 50% cancel. "
                       f"Measured directly on P(AI), the calibration error is **{b['ece_classwise']:.3f}**: on these unseen "
                       f"generators the model's average P(AI) was {pct(b['mean_p_fake'])} while {pct(b['share_fake'])} of "
                       "the images were AI-generated. Mid-range scores therefore understate the chance of AI generation "
                       "(see the calibration chart).", icon="⚠️")
        vd = card.get("verdicts")
        if vd:
            st.markdown(f"**With the inconclusive band (as this app reports):** {pct(vd['inconclusive_share'])} of benchmark "
                        f"images were *inconclusive*; the decided answers were right {pct(vd['decided_accuracy'])} of the "
                        f"time. AI images: {pct(vd['ai_images']['ai'])} called AI, {pct(vd['ai_images']['inconclusive'])} "
                        f"inconclusive, {pct(vd['ai_images']['real'])} called real. Real photos: "
                        f"{pct(vd['real_images']['real'])} called real, {pct(vd['real_images']['inconclusive'])} inconclusive, "
                        f"{pct(vd['real_images']['ai'])} called AI.")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**AI images caught, by generator family**")
            fam = pd.DataFrame([dict(family=k, caught=v["recall"], n=v["n"]) for k, v in card["per_family"].items()])
            bars = alt.Chart(fam).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("caught:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="detected"),
                y=alt.Y("family:N", sort="x", title=None),
                color=alt.condition(alt.datum.caught < 0.6, alt.value("#dc2626"), alt.value("#6366f1")),
                tooltip=["family", alt.Tooltip("caught:Q", format=".1%"), "n"])
            rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="x:Q")
            st.altair_chart(bars + rule, width="stretch")
        with g2:
            st.markdown("**Real photos kept real, by source**")
            src = pd.DataFrame([dict(source=k, kept=v["specificity"], n=v["n"]) for k, v in card["per_real_source"].items()])
            st.altair_chart(alt.Chart(src).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("kept:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="kept real"),
                y=alt.Y("source:N", sort="x", title=None),
                color=alt.condition(alt.datum.kept < 0.8, alt.value("#d97706"), alt.value("#16a34a")),
                tooltip=["source", alt.Tooltip("kept:Q", format=".1%"), "n"]), width="stretch")
        g3, g4 = st.columns(2)
        with g3:
            st.markdown("**Calibration: predicted vs. observed**")
            rb = pd.DataFrame([dict(predicted=(r["lo"] + r["hi"]) / 2, observed=r["fake_rate"], n=r["n"])
                               for r in card.get("reliability_bins", [])])
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 3], color="gray").encode(
                x="x:Q", y="y:Q")
            pts = alt.Chart(rb).mark_line(point=True, color="#6366f1").encode(
                x=alt.X("predicted:Q", axis=alt.Axis(format="%"), title="predicted P(AI), bin centre"),
                y=alt.Y("observed:Q", axis=alt.Axis(format="%"), title="share actually AI"),
                tooltip=[alt.Tooltip("predicted:Q", format=".0%"), alt.Tooltip("observed:Q", format=".1%"), "n"])
            st.altair_chart(diag + pts, width="stretch")
            st.caption("Points on the dashed diagonal = perfectly calibrated.")
        with g4:
            st.markdown("**Robustness: same images, degraded**")
            names = {"clean": "original", "jpeg75": "JPEG quality 75", "jpeg50": "JPEG quality 50", "half": "half size",
                     "quarter": "quarter size", "blur1": "Gaussian blur", "gray": "grayscale",
                     "social": "social-media style (1080 px + JPEG 70)"}
            st.dataframe([{"condition": names.get(k, k), "ROC-AUC": round(v["auc"], 3),
                           "balanced accuracy": pct(v["balanced_acc"], 1)} for k, v in card["robustness"].items()],
                         hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- 6. method & limits
with tab_method:
    st.markdown("""
**How the detector works**
1. The image is decoded to RGB (orientation, transparency and grayscale handled consistently).
2. Up to nine 256x256 regions are taken at **native resolution** - no resizing, because resizing erases the fine pixel
   statistics generators leave behind. Images smaller than 256 px are enlarged and flagged.
3. An **EfficientNet-B0** network scores each region (and its mirror image). Region log-odds are averaged.
4. The average is divided by a **temperature** fitted on a separate calibration set, so that "70%" means roughly
   "AI-generated 7 times in 10" on data like the calibration set. On generators it has never seen, the scores drift
   toward "real" (see *Reliability*).
5. An **inconclusive band** turns the probability into a verdict only when it is clear enough.
6. **Grad-CAM** maps the gradient of the model's own output back onto each region to show what influenced it.
""")
    st.markdown("**Known limits**")
    lim = list((card or {}).get("limitations", []))
    lim += ["Detects fully AI-generated images. It is not trained on face swaps, inpainting or other edits of real photos.",
            "Screenshots, heavy filters, very small images and re-uploads through social media weaken the signal."]
    st.markdown("\n".join(f"- {x}" for x in lim))
    st.caption("Full methodology, datasets, licences and results: README.md and the docs/ folder of the repository.")

# ----------------------------------------------------------------------------- 7. technical
with tab_tech:
    st.markdown(f"- **Model**: `{res.model_id}` ({res.kind})")
    st.markdown(f"- **Device**: {detector.device()}")
    st.markdown(f"- **Temperature**: {res.temperature:.3f} (fitted on a held-out calibration split, not on the benchmark)")
    if res.band:
        st.markdown(f"- **Inconclusive band on P(real)**: {res.band[0]:.3f} - {res.band[1]:.3f}")
    if res.tile_real:
        st.markdown(f"- **Regions scored**: {len(res.tile_real)} x 256 px on a {res.prepared_size[0]}x{res.prepared_size[1]} "
                    "px working image, with horizontal-flip averaging")
    st.markdown(f"- **Time**: analysis {t_analyze:.2f}s, attention map {t_explain:.2f}s (first run includes model loading)")
    st.markdown(f"- **File SHA-256**: `{hashlib.sha256(data).hexdigest()[:16]}...`")
    st.caption("`python predict.py <image> --json` runs the same code path and returns the same numbers.")

st.divider()
st.caption("Research prototype · results are probabilistic and can be wrong · never use as sole evidence about a person or "
           "an image · training and benchmark data are licensed for non-commercial research only.")
