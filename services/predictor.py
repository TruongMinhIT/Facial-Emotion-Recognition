import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown  # <-- Thêm thư viện tải file

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "ResNet50_Emotion_Augmentation_Attention.keras",
)

# ID Google Drive của bạn
DRIVE_FILE_ID = "1aM4iLTqMu4Qd7CpYM42_oCnSbPQyFh4W"

@st.cache_resource
def get_model():
    # Tạo thư mục models nếu chưa có
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # BƯỚC 1: Xóa cái file "dỏm" (file Git LFS hoặc file HTML tải nhầm dung lượng < 1MB)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 1000000:
        os.remove(MODEL_PATH)

    # BƯỚC 2: Tải file xịn từ Google Drive
    if not os.path.exists(MODEL_PATH):
        st.info("Đang tải trọng số AI từ Google Drive xuống máy chủ (Bỏ qua quét virus), vui lòng đợi...")
        # Dùng trực tiếp tham số id= để gdown tự động vượt qua trang cảnh báo virus của Google
        gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
        st.success("Tải mô hình thành công!")

    # BƯỚC 3: Load model chuẩn
    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "preprocess_input": preprocess_input
        }
    )

    return model

def predict_emotion(tensor):

    model = get_model()

    preds = model.predict(
        tensor,
        verbose=0
    )[0]

    idx = int(np.argmax(preds))

    return EMOTION_LABELS[idx], preds

def draw_results(image, x, y, w, h, label):

    color = (0, 255, 0)

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        color,
        2,
    )

    cv2.rectangle(
        image,
        (x, y - 30),
        (x + w, y),
        color,
        -1,
    )

    cv2.putText(
        image,
        label,
        (x + 5, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )

    return image