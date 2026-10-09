"""Tests for the shared detector, the file diagnostics, the demo samples and app.py / predict.py parity.

Run from the repo root:  python -m pytest tests -q
"""
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import detector  # noqa: E402
from detector import forensics  # noqa: E402
from detector.paths import MODELS, SAMPLES  # noqa: E402

pytestmark = pytest.mark.skipif(not detector.available(), reason="no fine-tuned checkpoint in models/")
FACE = "candidate_A_mild.pt"                       # face-crop models; the active default is a general model
ASSETS = os.path.join(ROOT, "tests", "assets")
IMAGES = sorted(glob.glob(os.path.join(ASSETS, "*")))
REAL_20 = os.path.join(ASSETS, "real_20.png")
TABS = ["Explanation", "Analysis breakdown", "File diagnostics", "Reliability", "Method & limits", "Technical"]


# ---------------------------------------------------------------- decoding / color handling
def test_to_rgb_handles_all_common_modes():
    base = np.random.RandomState(0).randint(0, 255, (40, 50, 3), dtype=np.uint8)
    cases = {
        "L": Image.fromarray(base[..., 0]),
        "LA": Image.fromarray(base[..., 0]).convert("LA"),
        "RGBA": Image.fromarray(base).convert("RGBA"),
        "P": Image.fromarray(base).convert("P"),
        "CMYK": Image.fromarray(base).convert("CMYK"),
        "I;16": Image.fromarray(base[..., 0].astype(np.uint16) * 257),
        "1": Image.fromarray(base[..., 0]).convert("1"),
    }
    for mode, im in cases.items():
        out = detector.to_rgb(im)
        assert out.mode == "RGB" and out.size == im.size, mode


def test_grayscale_stays_gray_and_opaque_alpha_is_dropped():
    g = Image.fromarray(np.random.RandomState(1).randint(0, 255, (30, 30), dtype=np.uint8))
    a = np.asarray(detector.to_rgb(g))
    assert (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()
    la = g.convert("LA")                      # fully opaque, like the grayscale PNGs in the training data
    assert (np.asarray(detector.to_rgb(la)) == a).all()


def test_transparent_pixels_are_composited_on_white():
    rgba = Image.new("RGBA", (8, 8), (10, 20, 30, 0))
    assert tuple(np.asarray(detector.to_rgb(rgba))[0, 0]) == (255, 255, 255)


# ---------------------------------------------------------------- shared detector behaviour
def test_analyze_is_deterministic_and_reports_model():
    r1 = detector.analyze(IMAGES[0])
    r2 = detector.analyze(IMAGES[0])
    assert r1.real_prob == r2.real_prob
    assert abs(r1.real_prob + r1.fake_prob - 1.0) < 1e-9
    assert r1.model_id.startswith("efficientnet_b0:") and "@" in r1.model_id
    assert r1.crop_policy in ("none", "face")


def test_explicit_crop_overrides_policy():
    assert detector.analyze(IMAGES[0], crop="none", model=FACE).crop_policy == "none"
    assert detector.analyze(IMAGES[0], crop="face", model=FACE).crop_policy == "face"
    with pytest.raises(ValueError):
        detector.analyze(IMAGES[0], crop="bogus", model=FACE)


def test_legacy_backend_flag_disables_detector(monkeypatch):
    monkeypatch.setenv("DETECTOR_BACKEND", "legacy")
    assert detector.available() is False


def test_model_selection_env_var(monkeypatch):
    monkeypatch.setenv("DETECTOR_MODEL", "does_not_exist.pt")
    assert detector.available() is False


def test_device_override_forces_cpu(monkeypatch):
    monkeypatch.setenv("DETECTOR_DEVICE", "cpu")
    assert detector.device() == "cpu"


def test_list_models_puts_general_models_first_and_cards_load():
    names = detector.list_models()
    assert names and all(os.path.isfile(os.path.join(MODELS, n)) for n in names)
    general = [n for n in names if n.startswith("general_")]
    assert names[:len(general)] == general
    if os.path.isfile(os.path.join(MODELS, "general_native_b.card.json")):
        card = detector.load_model_card("general_native_b.pt")
        assert {"benchmark", "per_family", "robustness", "reliability_bins"} <= set(card)
    assert detector.load_model_card("does_not_exist.pt") is None


# ---------------------------------------------------------------- demo samples and file diagnostics
def _manifest():
    with open(os.path.join(SAMPLES, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)["samples"]


def test_sample_manifest_matches_files_and_records_provenance():
    items = _manifest()
    assert items
    for s in items:
        with open(os.path.join(SAMPLES, s["file"]), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == s["sha256"], s["file"]
        assert s["label"] in ("ai_generated", "real") and s["label_verified"] is True
        assert s["used_in_training"] is False and s["license"] and s["label_source"] and s["display_name"]


def test_forensics_outputs_have_expected_shapes():
    path = os.path.join(SAMPLES, _manifest()[0]["file"])
    with open(path, "rb") as f:
        data = f.read()
    info = forensics.file_info(data)
    assert info["bytes"] == len(data) and info["size"][0] > 0 and info["format"]
    pil = detector.load_rgb(path)
    ela = forensics.ela(pil)
    assert ela.mode == "RGB" and max(ela.size) <= 1024
    spec = forensics.spectrum(pil)
    assert spec.shape == (256, 256, 3) and spec.dtype == np.uint8
    base = np.full((20, 30, 3), 100, np.uint8)
    heat = np.zeros((20, 30), np.float32)
    seen = np.zeros((20, 30), bool)
    seen[:, :15] = True
    out = forensics.blend_heatmap(base, heat, seen, 0.0)
    assert out.shape == base.shape
    assert (out[:, 15:] == 35).all()                     # unexamined pixels are dimmed to 35 %
    assert (out[:, :15] == 100).all()                    # opacity 0 leaves examined pixels untouched


def test_uploads_only_reach_the_advertised_decoders():
    import io
    from PIL import UnidentifiedImageError

    def encode(fmt, **kw):
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (90, 120, 150)).save(buf, fmt, **kw)
        return buf.getvalue()
    for fmt in ("JPEG", "PNG", "WEBP", "BMP", "TIFF"):
        assert forensics.file_info(encode(fmt))["size"] == (64, 48), fmt
    mpo = encode("MPO", save_all=True, append_images=[Image.new("RGB", (64, 48))])      # multi-picture phone JPEG
    assert forensics.open_upload(mpo).size == (64, 48)
    with pytest.raises(UnidentifiedImageError):                                       # real image, but not offered
        forensics.open_upload(encode("GIF"))


# ---------------------------------------------------------------- predict.py == detector.analyze
def _cli(path, *extra, cwd=ROOT):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "predict.py"), path, "--json", *extra],
                         capture_output=True, text=True, cwd=cwd, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def synthetic_images(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    rng = np.random.RandomState(7)
    paths = []
    img = rng.randint(0, 255, (200, 160, 3), dtype=np.uint8)
    for name, im in {
        "gray_la.png": Image.fromarray(img[..., 0]).convert("LA"),
        "rgba.png": Image.fromarray(img).convert("RGBA"),
        "rgb.jpg": Image.fromarray(img),
        "palette.png": Image.fromarray(img).convert("P"),
    }.items():
        p = str(d / name)
        im.save(p)
        paths.append(p)
    return paths


def test_cli_matches_detector_on_repo_and_synthetic_images(synthetic_images):
    for path in IMAGES[:4] + synthetic_images:
        ref = detector.analyze(path)
        cli = _cli(path)
        assert cli["model_id"] == ref.model_id, path
        assert cli["crop_policy"] == ref.crop_policy, path
        assert cli["real_prob"] == pytest.approx(ref.real_prob, abs=1e-4), path


def test_cli_works_from_another_working_directory(tmp_path):
    ref = detector.analyze(REAL_20)
    assert _cli(REAL_20, cwd=str(tmp_path))["real_prob"] == pytest.approx(ref.real_prob, abs=1e-4)


def test_cli_crop_flag_matches_detector():
    for crop in ("none", "face"):
        ref = detector.analyze(IMAGES[0], crop=crop, model=FACE)
        assert _cli(IMAGES[0], "--crop", crop, "--model", FACE)["real_prob"] == pytest.approx(ref.real_prob, abs=1e-4)


# ---------------------------------------------------------------- app.py == detector.analyze
def _general_models():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(MODELS, "general_*.pt"))
                  if os.path.exists(p[:-3] + ".meta.json"))


def _run_app(sample_file, model=None):
    """Open the app, optionally pick a detector in the sidebar, click a demo sample, return (AppTest, headline %)."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    if model:
        at.selectbox[0].select(model).run()
    at.button(key=f"sample_{sample_file}").click().run()
    assert not at.exception, at.exception
    headline = [m.value for m in at.markdown if 'class="score"' in str(m.value)]
    assert headline, "prediction card missing"
    shown = float(headline[0].split('class="score">')[1].split("%")[0]) / 100
    return at, shown


def test_app_sample_buttons_match_detector_and_never_load_swin(monkeypatch):
    try:
        import transformers
    except ImportError:
        transformers = None
    if transformers is not None:
        def boom(*a, **k):
            raise AssertionError("Swin / transformers.pipeline must not be loaded on the fine-tuned path")
        monkeypatch.setattr(transformers, "pipeline", boom)
    for s in _manifest()[:2]:
        at, shown = _run_app(s["file"])
        ref = detector.analyze(os.path.join(SAMPLES, s["file"]))
        assert shown == pytest.approx(ref.fake_prob, abs=0.006), s["file"]      # the app prints whole percent
        page = " ".join(str(getattr(e, "value", "")) for grp in (at.markdown, at.caption, at.text, at.metric) for e in grp)
        assert ref.model_id in page, "app must report which model produced the prediction"
        assert "Dataset label" in page, "the demo sample's ground truth must be revealed after the analysis"


@pytest.mark.skipif(not _general_models(), reason="no finalized general_* checkpoint")
def test_app_general_model_matches_detector_and_shows_the_evidence_sections():
    model = _general_models()[-1]
    s = _manifest()[1]
    at, shown = _run_app(s["file"], model)
    ref = detector.analyze(os.path.join(SAMPLES, s["file"]), model=model)
    assert ref.kind == "general"
    assert shown == pytest.approx(ref.fake_prob, abs=0.006)
    assert [t.label for t in at.tabs] == TABS
    page = " ".join(str(getattr(e, "value", "")) for e in at.markdown)
    assert "What the heat map is - and is not" in page and "What this means" in page
    verdict = {"likely_fake": "Likely AI-generated", "likely_real": "Likely real", "inconclusive": "Inconclusive"}[ref.verdict]
    assert verdict in page


@pytest.mark.skipif(not _general_models(), reason="no finalized general_* checkpoint")
def test_app_without_model_card_says_unvalidated():
    from streamlit.testing.v1 import AppTest
    model = _general_models()[-1]
    card = os.path.join(MODELS, f"{os.path.splitext(model)[0]}.card.json")
    hidden = None
    if os.path.exists(card):
        hidden = card + ".hidden"
        os.replace(card, hidden)
    try:
        at = AppTest.from_file("app.py", default_timeout=300)
        at.run()
        at.selectbox[0].select(model).run()
        at.button(key=f"sample_{_manifest()[0]['file']}").click().run()
        assert not at.exception, at.exception
        assert any("No measured model card" in str(w.value) for w in at.warning)
    finally:
        if hidden:
            os.replace(hidden, card)


def test_app_reset_returns_to_the_upload_screen():
    at, _ = _run_app(_manifest()[0]["file"])
    at.button(key="reset").click().run()
    assert not at.exception, at.exception
    assert not [m for m in at.markdown if 'class="score"' in str(m.value)]
    assert any(b.key and b.key.startswith("sample_") for b in at.button)


# ---------------------------------------------------------------- general-model inference
@pytest.mark.skipif(not _general_models(), reason="no finalized general_* checkpoint")
def test_general_analyze_is_deterministic_and_tiles_stay_inside_the_image():
    model = _general_models()[-1]
    rng = np.random.RandomState(3)
    for size in ((640, 480), (300, 1000), (257, 259), (200, 180)):
        im = Image.fromarray(rng.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8))
        r1, r2 = detector.analyze(im, model=model), detector.analyze(im, model=model)
        assert r1.real_prob == r2.real_prob and abs(r1.real_prob + r1.fake_prob - 1) < 1e-9
        w, h = r1.prepared_size
        for x0, y0, x1, y1 in r1.boxes:
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        assert r1.verdict in ("likely_fake", "likely_real", "inconclusive")
    small = detector.analyze(Image.fromarray(rng.randint(0, 255, (120, 120, 3), dtype=np.uint8)), model=model)
    assert "low_res" in [c for c, _ in small.flags]


@pytest.mark.skipif(not _general_models(), reason="no finalized general_* checkpoint")
def test_verdict_follows_the_decision_band():
    model = _general_models()[-1]
    r = detector.analyze(REAL_20, model=model)
    lo, hi = r.band
    expected = "likely_fake" if r.real_prob < lo else "likely_real" if r.real_prob > hi else "inconclusive"
    assert r.verdict == expected


@pytest.mark.skipif(not _general_models(), reason="no finalized general_* checkpoint")
def test_cli_matches_detector_for_general_model():
    model = _general_models()[-1]
    for path in IMAGES[:3]:
        ref = detector.analyze(path, model=model)
        cli = _cli(path, "--model", model)
        assert cli["real_prob"] == pytest.approx(ref.real_prob, abs=1e-4), path
        assert cli["verdict"] == ref.verdict and cli["kind"] == "general"


@pytest.mark.skipif(not _general_models(), reason="no finalized general_* checkpoint")
def test_explanation_is_tied_to_the_models_own_output():
    model = _general_models()[-1]
    im = Image.open(REAL_20)
    e1 = detector.explain(im, model=model)
    e2 = detector.explain(im, model=model)
    assert e1.layer and np.array_equal(e1.overlay, e2.overlay)                 # deterministic
    flipped = "fake" if e1.evidence_for == "real" else "real"
    e3 = detector.explain(im, model=model, evidence_for=flipped)
    assert not np.array_equal(e1.overlay, e3.overlay)                          # map depends on the class being explained
