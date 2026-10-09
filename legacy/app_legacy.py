import os
import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import streamlit as st
import detector  # fine-tuned PyTorch detector (preferred); TensorFlow is only a legacy fallback

# Reduce TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================

st.set_page_config(
    page_title="Universal Deepfake & AI Image Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom modern CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .badge-info {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin-right: 6px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS & CACHED MODEL LOADERS
# ==========================================

MODEL_PATH = r"models\best_xception.keras"
IMG_SIZE = (299, 299)

USE_FT = detector.available()   # False -> legacy Keras/Swin path (rollback: DETECTOR_BACKEND=legacy)


@st.cache_resource
def load_universal_xception():
    """Legacy rollback only: loads the original Keras Xception detector."""
    import tensorflow as tf  # legacy fallback only
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_swin_transformer():
    """Loads the Swin Transformer AI detector from Hugging Face."""
    try:
        from transformers import pipeline
        return pipeline("image-classification", model="umm-maybe/AI-image-detector", device="cpu")
    except Exception as e:
        st.warning(f"Swin Transformer engine unavailable: {e}")
        return None

# ==========================================
# FORENSICS & PREPROCESSING UTILITIES
# ==========================================

def detect_and_crop_face(pil_image):
    """
    Detects faces using OpenCV Haar Cascade.
    Returns: (cropped_face_pil, face_box_coords, annotated_pil_image)
    """
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    if len(faces) == 0:
        return pil_image, None, pil_image
    
    # Pick the largest detected face
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces_sorted[0]
    
    # Add 15% margin padding
    pad_w = int(w * 0.15)
    pad_h = int(h * 0.15)
    
    img_h, img_w = img_cv.shape[:2]
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)
    
    face_crop = pil_image.crop((x1, y1, x2, y2))
    
    # Draw bounding box on a copy
    annotated = pil_image.copy()
    annotated_cv = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
    cv2.rectangle(annotated_cv, (x1, y1), (x2, y2), (0, 255, 128), max(2, int(img_w / 300)))
    annotated = Image.fromarray(cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB))
    
    return face_crop, (x1, y1, x2, y2), annotated

def compute_fft_spectrum(pil_img):
    """Computes 2D Fourier Transform power spectrum heatmap."""
    gray = np.array(pil_img.convert("L").resize((256, 256)))
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))

def compute_ela(pil_img, quality=90):
    """Computes Error Level Analysis (JPEG compression difference)."""
    buf = io.BytesIO()
    pil_img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf)
    
    ela = ImageChops.difference(pil_img.convert("RGB"), compressed.convert("RGB"))
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max(max_diff, 1)
    return ImageEnhance.Brightness(ela).enhance(scale)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    if USE_FT:
        engine_choice = st.selectbox(
            "Detection Engine",
            ["Fine-tuned EfficientNet-B0 (active)"],
            index=0,
            help="Swin was removed from the active path: it scored near chance (AUC 0.58) in the audit."
        )
    else:
        engine_choice = st.selectbox(
            "Detection Engine",
            [
                "Dual-Engine Consensus (Highest Precision)",
                "Universal XceptionNet (Fast & Robust)",
                "Swin Vision Transformer (Modern AI Specialist)"
            ],
            index=0,
            help="Legacy rollback engines (fine-tuned checkpoint not available)."
        )

    st.divider()

    st.subheader("Face Analysis")
    DEFAULT_CROP_ON = detector.default_crop_policy() == "face" if USE_FT else True
    enable_face_crop = st.checkbox(
        "Auto-Crop Detected Face",
        value=DEFAULT_CROP_ON,
        help="Default follows the active model's validated crop policy (same as predict.py). "
             "Changing it overrides that policy."
    )
    
    st.divider()
    
    threshold = st.slider(
        "Classification Threshold",
        min_value=0.30,
        max_value=0.70,
        value=0.50,
        step=0.05,
        help="Images with Real Probability below this threshold are flagged as Synthetic/Fake."
    )
    
    st.divider()
    st.markdown("""
    **Training data (what was actually validated):**
    - StyleGAN faces (1M-fake-faces) vs. FFHQ photos
    - ~800 prompt-generated faces (generator unknown) vs. small real set

    Other generators (e.g. Gemini, Midjourney, Flux) have **not** been independently evaluated.
    """)

# ==========================================
# MAIN INTERFACE
# ==========================================

st.markdown('<div class="main-header">🛡️ Universal Deepfake & AI Image Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Estimate whether a face image is real or AI-generated / deepfaked, with FFT and ELA forensic views. Probabilities are calibrated on in-distribution data only.</div>', unsafe_allow_html=True)

# Badges
st.markdown("""
<div>
    <span class="badge-info">🧠 Fine-tuned EfficientNet-B0</span>
    <span class="badge-info">🔍 2D FFT & ELA Forensics</span>
    <span class="badge-info">📏 Temperature-calibrated</span>
</div>
""", unsafe_allow_html=True)

# Sample Quick Loader
st.write("---")
cols_quick = st.columns([1, 1, 1, 3])
load_sample = None

if cols_quick[0].button("📸 Sample Real"):
    load_sample = r"test_images\face_real.jpg"
if cols_quick[1].button("🤖 Sample Deepfake"):
    load_sample = r"test_images\face_fake.jpg"
if cols_quick[2].button("✨ Sample AI Face"):
    ai_samples = [
        r"data\ai_faces\AI-face-detection-Dataset\AI\0_20241130135552_A_close-up_shot_of_a_middle-aged_genderqueer_perso.jpg",
        r"data\ai_faces\AI-face-detection-Dataset\AI\100_20241130141829_A_close-up_shot_of_a_middle-aged_male_person_of_Na.jpg"
    ]
    for s in ai_samples:
        if os.path.exists(s):
            load_sample = s
            break

uploaded_file = st.file_uploader(
    "Upload an image to inspect (JPEG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"]
)

target_image = None
if uploaded_file is not None:
    target_image = detector.to_rgb(Image.open(uploaded_file))
elif load_sample and os.path.exists(load_sample):
    target_image = detector.load_rgb(load_sample)
    st.info(f"Loaded sample: `{os.path.basename(load_sample)}`")

# ==========================================
# INFERENCE & RESULTS
# ==========================================

if target_image is not None:
    if USE_FT:
        # Shared path: identical decoding / crop / resize / calibration to predict.py (detector.analyze)
        crop_mode = "auto" if enable_face_crop == DEFAULT_CROP_ON else ("face" if enable_face_crop else "none")
        ft = detector.analyze(target_image, crop=crop_mode)
        eval_image = ft.region
        annotated_img = target_image
        if ft.box is not None:
            ann = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
            cv2.rectangle(ann, ft.box[:2], ft.box[2:], (0, 255, 128), max(2, int(target_image.size[0] / 300)))
            annotated_img = Image.fromarray(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
        crop_status = (f"Crop policy: {ft.crop_policy}" +
                       ("" if ft.face_found is None else (" - face cropped with margin." if ft.face_found
                                                         else " - no face found, full image analyzed.")))
    else:
        # Legacy rollback path
        face_crop, face_coords, annotated_img = detect_and_crop_face(target_image)
        if enable_face_crop and face_coords is not None:
            eval_image = face_crop
            crop_status = "Detected face cropped with margin."
        else:
            eval_image = target_image
            crop_status = "Full image analyzed (no face detected or crop disabled)."

    # Layout: Image Display
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Uploaded Image")
        st.image(annotated_img, use_container_width=True)
        st.caption(f"Status: {crop_status}")
        
    with col_right:
        st.subheader("Analyzed Region")
        st.image(eval_image, use_container_width=True)
        st.caption(f"Dimensions: {eval_image.size[0]} x {eval_image.size[1]} px")

    # Run Prediction
    with st.spinner("Analyzing neural texture and frequency signatures..."):
        p_xc_fake, p_xc_real = None, None
        p_swin_fake, p_swin_real = None, None

        # Universal Xception (legacy rollback only; the fine-tuned path above never loads it or Swin)
        if not USE_FT and ("Xception" in engine_choice or "Consensus" in engine_choice):
            xc_model = load_universal_xception()
            if xc_model:
                import tensorflow as tf
                img_xc = eval_image.resize(IMG_SIZE)
                arr = np.expand_dims(np.array(img_xc, dtype=np.float32), 0)
                arr = tf.keras.applications.xception.preprocess_input(arr)
                p_xc_real = float(xc_model.predict(arr, verbose=0)[0][0])
                p_xc_fake = 1.0 - p_xc_real
        
        # Swin Transformer (legacy rollback only)
        if not USE_FT and ("Swin" in engine_choice or "Consensus" in engine_choice):
            swin_pipe = load_swin_transformer()
            if swin_pipe:
                res = swin_pipe(eval_image)
                p_swin_fake = next(item["score"] for item in res if item["label"] == "artificial")
                p_swin_real = next(item["score"] for item in res if item["label"] == "human")
        
        # Calculate Final Probabilities based on selected engine
        if USE_FT:
            # Single calibrated EfficientNet-B0 (audit: Swin AUC 0.58 and it degraded calibration)
            p_xc_real, p_xc_fake = ft.real_prob, ft.fake_prob
            final_real_prob, final_fake_prob = ft.real_prob, ft.fake_prob
            engine_label = f"EfficientNet-B0 ({ft.model_id})"
        elif "Consensus" in engine_choice and p_swin_fake is not None and p_xc_fake is not None:
            final_fake_prob = 0.55 * p_swin_fake + 0.45 * p_xc_fake
            final_real_prob = 1.0 - final_fake_prob
            engine_label = "Dual-Engine Consensus (Swin ViT + Universal Xception)"
        elif "Swin" in engine_choice and p_swin_fake is not None:
            final_fake_prob = p_swin_fake
            final_real_prob = p_swin_real
            engine_label = "Swin Vision Transformer"
        else:
            final_fake_prob = p_xc_fake if p_xc_fake is not None else 0.5
            final_real_prob = p_xc_real if p_xc_real is not None else 0.5
            engine_label = "Universal XceptionNet"

    # Verdict
    is_real = final_real_prob >= threshold
    verdict = "REAL (AUTHENTIC)" if is_real else "FAKE (AI / DEEPFAKE)"
    
    st.write("---")
    st.subheader("Prediction Result")
    
    if is_real:
        st.success(f"### 🟢 {verdict}")
        st.markdown(f"**Estimate:** {final_real_prob*100:.1f}% probability that this image is authentic.")
    else:
        st.error(f"### 🔴 {verdict}")
        st.markdown(f"**Estimate:** {final_fake_prob*100:.1f}% probability that this image is synthetic or manipulated.")
    st.caption("Calibrated on in-distribution test data only; not a guarantee for unseen generators, "
               "heavy compression, blur or very low resolution.")

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Authentic Probability", f"{final_real_prob * 100:.2f}%")
    m2.metric("Synthetic / Fake Probability", f"{final_fake_prob * 100:.2f}%")
    m3.metric("Active Engine", engine_label.split(" (")[0])

    # Probability Bars
    st.write("**Real Probability:**")
    st.progress(float(min(max(final_real_prob, 0.0), 1.0)))
    
    st.write("**Fake / AI Probability:**")
    st.progress(float(min(max(final_fake_prob, 0.0), 1.0)))

    # ======================================
    # FORENSIC EXPLAINABILITY TABS
    # ======================================
    st.write("---")
    st.subheader("🔬 Forensic Analysis Suite")
    
    tab_ensemble, tab_fft, tab_ela, tab_guide = st.tabs([
        "📊 Model Breakdown",
        "🌐 Frequency Spectrum (FFT)",
        "🔍 Error Level Analysis (ELA)",
        "📖 Artifact Guide"
    ])
    
    with tab_ensemble:
      if USE_FT:
        st.markdown("#### Active model")
        st.write(f"- Model: `{ft.model_id}`")
        st.write(f"- Crop policy applied: `{ft.crop_policy}`" +
                 ("" if ft.face_found is None else f" (face found: {ft.face_found})"))
        st.write(f"- Temperature: `{ft.temperature:.3f}`")
        st.write(f"- Real: `{ft.real_prob*100:.1f}%` | Fake: `{ft.fake_prob*100:.1f}%`")
        st.caption("Swin is not loaded or used: it scored near chance on held-out faces in the audit.")
      else:
        st.markdown("#### Individual Engine Assessment")
        e_col1, e_col2 = st.columns(2)
        
        with e_col1:
            st.markdown("**Universal XceptionNet (Deepfake & AI Faces)**")
            if p_xc_real is not None:
                st.write(f"- Real: `{p_xc_real*100:.1f}%` | Fake: `{p_xc_fake*100:.1f}%`")
                st.write(f"- Verdict: {'🟢 Real' if p_xc_real >= threshold else '🔴 Fake'}")
            else:
                st.write("*Not invoked for this selection.*")
                
        with e_col2:
            st.markdown("**Swin Vision Transformer (Modern Generative AI)**")
            if p_swin_real is not None:
                st.write(f"- Real: `{p_swin_real*100:.1f}%` | Fake: `{p_swin_fake*100:.1f}%`")
                st.write(f"- Verdict: {'🟢 Real' if p_swin_real >= threshold else '🔴 Fake'}")
            else:
                st.write("*Not invoked for this selection.*")

    with tab_fft:
        st.markdown("#### 2D Fourier Frequency Spectrum")
        st.markdown("""
        Natural camera sensors produce a smooth radial roll-off in the high-frequency Fourier spectrum.
        Generative models (Gemini / Imagen, Stable Diffusion, GANs) leave distinctive **high-frequency spectral spikes, checkerboards, or unnatural circular decay**.
        """)
        fft_view = compute_fft_spectrum(eval_image)
        st.image(fft_view, caption="2D Power Spectrum (Log Magnitude with Viridis Map)", width=350)

    with tab_ela:
        st.markdown("#### Error Level Analysis (ELA)")
        st.markdown("""
        ELA highlights compression rate variances across the image.
        In authentic camera JPEGs, error levels are uniformly distributed. In AI-generated faces or spliced deepfakes, manipulated regions often exhibit distinct brightness discrepancies.
        """)
        ela_view = compute_ela(eval_image)
        st.image(ela_view, caption="ELA Heatmap (Enhanced Difference at 90% Quality)", use_container_width=True)

    with tab_guide:
        st.markdown("""
        ### Known limits of this detector
        1. **Training coverage is narrow**:
           - It was trained on StyleGAN faces (1M-fake-faces) vs. FFHQ, plus ~800 prompt-generated faces of unknown generator.
           - Diffusion generators such as Gemini/Imagen, Midjourney, Flux or Stable Diffusion have not been independently evaluated.
        2. **Image quality matters**:
           - Heavy blur, strong down-scaling and grayscale conversion can push predictions toward "real" for the original model.
           - A high displayed probability is not a guarantee for unseen generators or degraded images.
        3. **Face crop**:
           - The crop policy is a per-model setting chosen on validation data; the sidebar toggle overrides it.
        """)

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption("Universal AI & Deepfake Detection Engine | CDAC Project")