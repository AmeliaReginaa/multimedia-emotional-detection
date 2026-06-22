import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import time
import os
from model import load_model

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stress Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL STYLE ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root gradient background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a1628 0%, #0d2247 30%, #0f3460 60%, #1565c0 100%) !important;
    min-height: 100vh;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Main content area ── */
[data-testid="stMainBlockContainer"] {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ── Global font ── */
* {
    font-family: 'Inter', sans-serif;
}

/* ── Hero section ── */
.hero-wrapper {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-bottom: 1rem;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 50px;
    padding: 0.3rem 1.1rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: #90caf9;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2rem, 5vw, 3.4rem);
    font-weight: 700;
    color: #ffffff;
    line-height: 1.15;
    margin: 0 0 0.75rem;
    letter-spacing: -0.02em;
}

.hero-title span {
    background: linear-gradient(90deg, #64b5f6, #42a5f5, #90caf9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.55);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 400;
}

/* ── Card ── */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 1.8rem;
    backdrop-filter: blur(12px);
    height: 100%;
}

.card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: #90caf9;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

/* ── Result box ── */
.result-box {
    border-radius: 16px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin: 1.2rem 0;
    position: relative;
    overflow: hidden;
}

.result-box.stress {
    background: linear-gradient(135deg, rgba(239,83,80,0.18) 0%, rgba(183,28,28,0.22) 100%);
    border: 1px solid rgba(239,83,80,0.4);
}

.result-box.no-stress {
    background: linear-gradient(135deg, rgba(38,166,154,0.18) 0%, rgba(0,105,92,0.22) 100%);
    border: 1px solid rgba(38,166,154,0.4);
}

.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0.3rem 0 0.1rem;
    letter-spacing: -0.01em;
}

.result-box.stress .result-label  { color: #ef9a9a; }
.result-box.no-stress .result-label { color: #80cbc4; }

.result-emoji {
    font-size: 2.4rem;
    line-height: 1;
    margin-bottom: 0.3rem;
}

.result-conf {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.5);
    margin-top: 0.2rem;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    margin: 0.8rem 0 0.4rem;
}

.conf-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5);
    margin-bottom: 0.3rem;
}

.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 50px;
    height: 8px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 0.6s ease;
}

.conf-bar-fill.stress   { background: linear-gradient(90deg, #e57373, #ef5350); }
.conf-bar-fill.nostress { background: linear-gradient(90deg, #4db6ac, #26a69a); }

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 2rem 0;
}

/* ── Info pill ── */
.info-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 50px;
    padding: 0.35rem 0.9rem;
    font-size: 0.74rem;
    color: rgba(255,255,255,0.6);
    margin: 0.25rem;
}

/* ── Streamlit widget overrides ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 2px dashed rgba(100,181,246,0.35) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span {
    color: rgba(255,255,255,0.55) !important;
}

div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1565c0, #1976d2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(21,101,192,0.4) !important;
    width: 100% !important;
}

div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #1976d2, #1e88e5) !important;
    box-shadow: 0 6px 20px rgba(21,101,192,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Radio buttons ── */
div[data-testid="stRadio"] label {
    color: rgba(255,255,255,0.75) !important;
}

div[data-testid="stRadio"] > div {
    gap: 0.5rem;
}

/* ── Checkbox ── */
div[data-testid="stCheckbox"] label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.88rem !important;
}

/* ── Image display ── */
[data-testid="stImage"] {
    border-radius: 14px;
    overflow: hidden;
}

/* ── Warning / info ── */
[data-testid="stWarning"], [data-testid="stInfo"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: rgba(255,255,255,0.7) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 0.5rem;
    color: rgba(255,255,255,0.25);
    font-size: 0.75rem;
    letter-spacing: 0.04em;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #64b5f6 !important;
}
</style>
""", unsafe_allow_html=True)


# ── PREPROCESSING ─────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

CLASS_NAMES = ["non_stress", "stress"]


# ── MODEL LOADER ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(weights_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = load_model(weights_path, device=device)
        return model, device
    except Exception as e:
        return None, str(e)


# ── INFERENCE ─────────────────────────────────────────────────────────────────
def predict(pil_img: Image.Image, model, device: str):
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx  = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_label, confidence, probs


# ── RENDER RESULT ─────────────────────────────────────────────────────────────
def render_result(label: str, confidence: float, probs):
    is_stress  = label == "stress"
    box_class  = "stress" if is_stress else "no-stress"
    emoji      = "😰" if is_stress else "😊"
    display    = "TERDETEKSI STRES" if is_stress else "TIDAK ADA STRES"

    st.markdown(f"""
    <div class="result-box {box_class}">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label">{display}</div>
        <div class="result-conf">Kepercayaan model: {confidence*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bars
    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probs)):
        bar_class = "stress" if cls == "stress" else "nostress"
        pct = prob * 100
        lbl = "Stres" if cls == "stress" else "Tidak Stres"
        st.markdown(f"""
        <div class="conf-bar-wrap">
            <div class="conf-bar-label">
                <span>{lbl}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill {bar_class}" style="width:{pct}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-badge">🧠 Deep Learning · CNN · FER Dataset</div>
        <h1 class="hero-title">Deteksi <span>Stres</span> Wajah</h1>
        <p class="hero-sub">Unggah foto atau gunakan kamera untuk mendeteksi tingkat stres melalui ekspresi wajah secara real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model path input ───────────────────────────────────────────────────────
    with st.expander("⚙️ Pengaturan Model", expanded=False):
        weights_path = st.text_input(
            "Path file model (.pth)",
            value="best_emotion_model.pth",
            help="Letakkan file .pth di folder yang sama dengan app.py, lalu masukkan nama filenya."
        )

    # ── Load model ─────────────────────────────────────────────────────────────
    model_loaded = False
    if weights_path and os.path.exists(weights_path):
        model, device = get_model(weights_path)
        if model is not None:
            model_loaded = True
            st.markdown(f"""
            <div style="text-align:center; margin-bottom:1rem;">
                <span class="info-pill">✅ Model aktif</span>
                <span class="info-pill">💻 {device.upper()}</span>
                <span class="info-pill">🏗️ EmotionCNNDeepFC</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"Gagal memuat model: {device}")
    else:
        st.markdown("""
        <div style="text-align:center; margin-bottom:1rem;">
            <span class="info-pill">⚠️ File model belum ditemukan — masukkan path yang benar di Pengaturan Model</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Mode tabs ──────────────────────────────────────────────────────────────
    mode = st.radio(
        "Pilih metode input:",
        ["📁  Upload Gambar", "📷  Kamera Webcam"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 1 — UPLOAD
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "📁  Upload Gambar":
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📂 Unggah Foto</div>', unsafe_allow_html=True)

            uploaded = st.file_uploader(
                "Pilih gambar wajah",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )

            show_processed = st.checkbox("Tampilkan gambar setelah preprocessing", value=False)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_result:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔍 Hasil Analisis</div>', unsafe_allow_html=True)

            if uploaded is not None:
                pil_img = Image.open(uploaded).convert("RGB")
                col_upload.image(pil_img, caption="Gambar yang diunggah", use_container_width=True)

                if show_processed:
                    gray = pil_img.convert("L").resize((48, 48))
                    col_upload.image(gray, caption="48×48 grayscale (input model)", width=150)

                if model_loaded:
                    with st.spinner("Menganalisis ekspresi wajah..."):
                        time.sleep(0.4)   # biar animasi spinner keliatan
                        label, conf, probs = predict(pil_img, model, device)
                    render_result(label, conf, probs)
                else:
                    st.info("Muat model terlebih dahulu untuk melihat hasil analisis.")
            else:
                st.markdown("""
                <div style="text-align:center; padding:3rem 0; color:rgba(255,255,255,0.25);">
                    <div style="font-size:3rem;">🖼️</div>
                    <div style="margin-top:0.5rem; font-size:0.85rem;">Belum ada gambar yang diunggah</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 2 — WEBCAM
    # ══════════════════════════════════════════════════════════════════════════
    else:
        col_cam, col_res = st.columns([1, 1], gap="large")

        with col_cam:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📷 Kamera Webcam</div>', unsafe_allow_html=True)

            cam_img = st.camera_input(
                "Ambil foto dari kamera",
                label_visibility="collapsed",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔍 Hasil Analisis</div>', unsafe_allow_html=True)

            if cam_img is not None:
                pil_img = Image.open(cam_img).convert("RGB")

                if model_loaded:
                    with st.spinner("Menganalisis ekspresi wajah..."):
                        time.sleep(0.3)
                        label, conf, probs = predict(pil_img, model, device)
                    render_result(label, conf, probs)
                else:
                    st.info("Muat model terlebih dahulu untuk melihat hasil analisis.")
            else:
                st.markdown("""
                <div style="text-align:center; padding:3rem 0; color:rgba(255,255,255,0.25);">
                    <div style="font-size:3rem;">📷</div>
                    <div style="margin-top:0.5rem; font-size:0.85rem;">Belum ada foto yang diambil</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Continuous capture hint
        if cam_img is not None and model_loaded:
            st.markdown("""
            <div style="text-align:center; margin-top:0.8rem;">
                <span class="info-pill">💡 Klik tombol kamera lagi untuk menganalisis frame baru</span>
            </div>
            """, unsafe_allow_html=True)

    # ── About section ──────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    with st.expander("ℹ️ Tentang Model"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("""
            **Arsitektur**
            - EmotionCNNDeepFC
            - 3 Convolutional Blocks
            - 5-layer Fully Connected
            - ~11.3 juta parameter
            """)
        with b:
            st.markdown("""
            **Dataset**
            - FER (Facial Expression Recognition)
            - 28.821 gambar training
            - 7.066 gambar validasi
            - 2 kelas: Stres / Tidak Stres
            """)
        with c:
            st.markdown("""
            **Performa (Validasi)**
            - Accuracy: 84.70%
            - F1-Score Stres: 0.83
            - F1-Score Non-Stres: 0.86
            - AUC-ROC: 0.9258
            """)

    # ── Footer ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        Stress Detection · CNN · Facial Expression Recognition · PyTorch
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
