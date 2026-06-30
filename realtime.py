import streamlit as st
import cv2
import av
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# =====================================================
# MODEL CNN
# =====================================================
class EmotionCNNDeepFC(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.30),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256*6*6,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.50),

            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.40),

            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =====================================================
# LOAD MODEL
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionCNNDeepFC()
model.load_state_dict(
    torch.load(
        "best_emotion_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

class_names = [
    "NON-STRESS",
    "STRESS"
]

# =====================================================
# TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

# =====================================================
# FACE DETECTOR
# =====================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =====================================================
# PREDICT IMAGE
# =====================================================
def predict_image(image):

    image_tensor = transform(
        np.array(image)
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_tensor)

        probs = torch.softmax(
            output,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )

    return (
        pred.item(),
        confidence.item(),
        probs.cpu().numpy()[0]
    )

# =====================================================
# STREAMLIT UI CONFIG
# =====================================================

st.set_page_config(
    page_title="CNN Stress Detection Dashboard",
    layout="wide"
)

# =====================================================
# CUSTOM CSS - BLUE GRADIENT THEME
# =====================================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0a1a3c 0%, #14306e 35%, #1f4e9c 70%, #2f6fd6 100%);
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.app-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: #ffffff;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: 0.5px;
}

.app-subtitle {
    text-align: center;
    color: #cfe0ff;
    font-size: 1rem;
    margin-bottom: 1.8rem;
    font-weight: 400;
}

.glass-card {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 20, 60, 0.35);
    height: 100%;
}

.section-label {
    color: #ffffff;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.result-stress {
    background: linear-gradient(135deg, #ff5f6d, #c0392b);
    color: white;
    padding: 0.9rem 1.2rem;
    border-radius: 14px;
    font-weight: 700;
    font-size: 1.2rem;
    text-align: center;
    box-shadow: 0 6px 18px rgba(192, 57, 43, 0.4);
    margin-bottom: 0.9rem;
}

.result-nonstress {
    background: linear-gradient(135deg, #00c896, #1ea1f1);
    color: white;
    padding: 0.9rem 1.2rem;
    border-radius: 14px;
    font-weight: 700;
    font-size: 1.2rem;
    text-align: center;
    box-shadow: 0 6px 18px rgba(30, 161, 241, 0.4);
    margin-bottom: 0.9rem;
}

.metric-box {
    background: rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.15);
}

.metric-box .label {
    color: #cfe0ff;
    font-size: 0.8rem;
    font-weight: 500;
}

.metric-box .value {
    color: #ffffff;
    font-size: 1.4rem;
    font-weight: 800;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.06);
    padding: 6px;
    border-radius: 14px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    color: #cfe0ff;
    font-weight: 600;
    padding: 8px 18px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2f6fd6, #1ea1f1) !important;
    color: white !important;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0.8rem;
    border: 1px dashed rgba(255,255,255,0.35);
}

[data-testid="stFileUploader"] label {
    color: #e6efff !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #1ea1f1, #00c896);
}

.info-banner {
    background: rgba(255,255,255,0.08);
    border-left: 4px solid #1ea1f1;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #e6efff;
    margin-bottom: 1rem;
}

.footer-card {
    background: rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 1.5rem;
    border: 1px solid rgba(255,255,255,0.15);
}

.footer-info-line {
    color: #e6efff !important;
    margin: 0.35rem 0 !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-title">🧠 CNN Stress Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Sistem multimedia untuk mendeteksi kondisi stress dan non-stress menggunakan CNN berbasis ekspresi wajah</div>',
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs([
    "📁 Upload Image",
    "🎥 Realtime Webcam"
])

# =====================================================
# VIDEO PROCESSOR
# =====================================================
class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            try:

                face_tensor = transform(
                    face
                ).unsqueeze(0).to(device)

                with torch.no_grad():

                    output = model(face_tensor)

                    probs = torch.softmax(
                        output,
                        dim=1
                    )

                    confidence, pred = torch.max(
                        probs,
                        dim=1
                    )

                label = class_names[
                    pred.item()
                ]

                conf = confidence.item() * 100

                if label == "STRESS":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(
                    img,
                    (x, y),
                    (x+w, y+h),
                    color,
                    2
                )

                cv2.putText(
                    img,
                    f"{label} {conf:.1f}%",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            except:
                pass

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# =====================================================
# TAB 1 - UPLOAD IMAGE
# =====================================================
with tab1:

    uploaded_file = st.file_uploader(
        "Upload Face Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns([1, 1.6], gap="large")

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">🖼️ Input Image</div>', unsafe_allow_html=True)
            st.image(image, width=260)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:

            pred, conf, probs = predict_image(image)

            label = class_names[pred]

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">📊 Detection Result</div>', unsafe_allow_html=True)

            if label == "STRESS":
                st.markdown(f'<div class="result-stress">🚨 {label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-nonstress">✅ {label}</div>', unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    f'<div class="metric-box"><div class="label">Confidence</div>'
                    f'<div class="value">{conf*100:.2f}%</div></div>',
                    unsafe_allow_html=True
                )
            with m2:
                st.markdown(
                    f'<div class="metric-box"><div class="label">Predicted Class</div>'
                    f'<div class="value">{label}</div></div>',
                    unsafe_allow_html=True
                )

            st.write("")
            st.progress(conf)

            st.markdown('<div class="section-label" style="margin-top:1rem;">📈 Probability Distribution</div>', unsafe_allow_html=True)

            st.bar_chart({
                "NON-STRESS": float(probs[0]),
                "STRESS": float(probs[1])
            })

            st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# TAB 2 - REALTIME WEBCAM
# =====================================================
with tab2:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🎥 Realtime Webcam Detection</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-banner">Arahkan wajah ke kamera. Hasil prediksi akan muncul langsung pada video.</div>',
        unsafe_allow_html=True
    )

    webrtc_streamer(
        key="stress-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# MODEL INFORMATION FOOTER
# =====================================================
st.markdown('<div class="footer-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📌 Model Information</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        '<div class="metric-box"><div class="label">Accuracy</div><div class="value">84.81%</div></div>',
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        '<div class="metric-box"><div class="label">Classes</div><div class="value">2</div></div>',
        unsafe_allow_html=True
    )

st.write("")
st.markdown(
    """
    <div class="footer-info-line">📁 Dataset : FER2013</div>
    <div class="footer-info-line">🧠 Model : EmotionCNNDeepFC</div>
    <div class="footer-info-line">🖼️ Input : 48x48 Grayscale</div>
    <div class="footer-info-line">🏷️ Classes : Stress / Non-Stress</div>
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)