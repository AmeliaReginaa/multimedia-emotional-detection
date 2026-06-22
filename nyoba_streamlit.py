import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="😊",
    layout="wide"
)

# ====================================
# CUSTOM CSS
# ====================================

st.markdown("""
<style>

.main {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #334155
    );
}

.title {
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:white;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
}

.card{
    background-color:#1e293b;
    padding:25px;
    border-radius:20px;
    box-shadow:0 0 15px rgba(0,0,0,0.3);
}

.result{
    text-align:center;
    font-size:35px;
    color:#38bdf8;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ====================================
# HEADER
# ====================================

st.markdown(
    "<div class='title'>😊 Emotion Detection AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Upload Image atau Gunakan Webcam untuk Mendeteksi Emosi</div>",
    unsafe_allow_html=True
)

st.write("")

# ====================================
# LOAD LABEL
# ====================================

with open("class_names.txt") as f:
    class_names = [x.strip() for x in f.readlines()]

# ====================================
# MODEL
# ====================================

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128*28*28,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,len(class_names))
        )

    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

model = EmotionCNN()

model.load_state_dict(
    torch.load(
        "best_emotion_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# ====================================
# TRANSFORM
# ====================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ====================================
# PREDICT FUNCTION
# ====================================

def predict_image(image):

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prob = torch.softmax(output, dim=1)

        confidence, pred = torch.max(prob,1)

    return (
        class_names[pred.item()],
        confidence.item()*100
    )

# ====================================
# MENU
# ====================================

tab1, tab2 = st.tabs([
    "📷 Upload Image",
    "🎥 Webcam"
])

# ====================================
# TAB 1
# ====================================

with tab1:

    uploaded = st.file_uploader(
        "Upload Foto",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        col1,col2 = st.columns(2)

        with col1:
            st.image(image,use_container_width=True)

        with col2:

            emotion, conf = predict_image(image)

            st.success("Prediksi Berhasil")

            st.markdown(
                f"<div class='result'>{emotion}</div>",
                unsafe_allow_html=True
            )

            st.metric(
                "Confidence",
                f"{conf:.2f}%"
            )

# ====================================
# TAB 2
# ====================================

with tab2:

    picture = st.camera_input(
        "Ambil Foto dari Webcam"
    )

    if picture:

        image = Image.open(picture).convert("RGB")

        st.image(image)

        emotion, conf = predict_image(image)

        st.success("Prediksi Berhasil")

        st.markdown(
            f"<div class='result'>{emotion}</div>",
            unsafe_allow_html=True
        )

        st.metric(
            "Confidence",
            f"{conf:.2f}%"
        )