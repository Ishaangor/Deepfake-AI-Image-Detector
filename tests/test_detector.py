"""Tests for the shared detector and for app.py / predict.py parity.

Run from the repo root:  python -m pytest tests -q
"""
import glob
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

pytestmark = pytest.mark.skipif(not detector.available(), reason="no fine-tuned checkpoint in models/")
FACE = "candidate_A_mild.pt"                       # face-crop models; the active default is a general model
IMAGES = sorted(glob.glob(os.path.join("test_images", "*")))


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


# ---------------------------------------------------------------- predict.py == detector.analyze
def _cli(path, *extra):
    out = subprocess.run([sys.executable, "predict.py", path, "--json", *extra], capture_output=True, text=True,
                         cwd=ROOT, timeout=300)
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


def test_cli_crop_flag_matches_detector():
    for crop in ("none", "face"):
        ref = detector.analyze(IMAGES[0], crop=crop, model=FACE)
        assert _cli(IMAGES[0], "--crop", crop, "--model", FACE)["real_prob"] == pytest.approx(ref.real_prob, abs=1e-4)


# ---------------------------------------------------------------- app.py == detector.analyze
def _general_models():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join("models", "general_*.pt"))
                  if os.path.exists(p[:-3] + ".meta.json"))


def _run_app(sample_file, model=None):
    """Open the app, optionally pick a detector in the sidebar, click a bundled sample, return (AppTest, headline %)."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    if model:
        at.selectbox[0].select(model).run()
    [b for b in at.button if b.label == sample_file][0].click().run()
    assert not at.exception, at.exception
    headline = [m.value for m in at.markdown if "calibrated probability" in str(m.value)]
    assert headline, "prediction card missing"
    shown = float(headline[0].split('class="score">')[1].split("%")[0]) / 100
    return at, shown


def test_app_sample_buttons_match_detector_and_never_load_swin(monkeypatch):
    import transformers

    def boom(*a, **k):
        raise AssertionError("Swin / transformers.pipeline must not be loaded on the fine-tuned path")
    monkeypatch.setattr(transformers, "pipeline", boom)
    for fname in ("face_real.jpg", "face_fake.jpg"):
        at, shown = _run_app(fname)
        ref = detector.analyze(os.path.join("test_images", fname))
        assert shown == pytest.approx(ref.fake_prob, abs=0.006), fname      # the app prints whole percent
        page = " ".join(str(getattr(e, "value", "")) for grp in (at.markdown, at.caption, at.text, at.metric) for e in grp)
        assert ref.model_id in page, "app must report which model produced the prediction"


@pytest.mark.skipif(not _general_models(), reason="no finalized general_* checkpoint")
def test_app_general_model_matches_detector_and_shows_the_evidence_sections():
    model = _general_models()[-1]
    at, shown = _run_app("real_20.png", model)
    ref = detector.analyze(os.path.join("test_images", "real_20.png"), model=model)
    assert ref.kind == "general"
    assert shown == pytest.approx(ref.fake_prob, abs=0.006)
    assert [t.label for t in at.tabs] == ["Visual evidence", "Forensic observations", "How reliable is this?",
                                          "Technical details"]
    page = " ".join(str(getattr(e, "value", "")) for e in at.markdown)
    assert "What this map is - and is not" in page and "Uncertainty" in page
    verdict = {"likely_fake": "Likely AI-generated", "likely_real": "Likely real", "inconclusive": "Inconclusive"}[ref.verdict]
    assert verdict in page


@pytest.mark.skipif(not _general_models(), reason="no finalized general_* checkpoint")
def test_app_without_model_card_says_unvalidated(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    model = _general_models()[-1]
    card = os.path.join("artifacts", f"model_card_{os.path.splitext(model)[0]}.json")
    hidden = None
    if os.path.exists(card):
        hidden = card + ".hidden"
        os.replace(card, hidden)
    try:
        at = AppTest.from_file("app.py", default_timeout=300)
        at.run()
        at.selectbox[0].select(model).run()
        [b for b in at.button if b.label == "real_20.png"][0].click().run()
        assert not at.exception, at.exception
        assert any("No measured model card" in str(w.value) for w in at.warning)
    finally:
        if hidden:
            os.replace(hidden, card)


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
    r = detector.analyze(os.path.join("test_images", "real_20.png"), model=model)
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
    im = Image.open(os.path.join("test_images", "real_20.png"))
    e1 = detector.explain(im, model=model)
    e2 = detector.explain(im, model=model)
    assert e1.layer and np.array_equal(e1.overlay, e2.overlay)                 # deterministic
    flipped = "fake" if e1.evidence_for == "real" else "real"
    e3 = detector.explain(im, model=model, evidence_for=flipped)
    assert not np.array_equal(e1.overlay, e3.overlay)                          # map depends on the class being explained
