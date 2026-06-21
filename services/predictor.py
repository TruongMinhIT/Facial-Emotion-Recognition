import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown  # Đã thêm thư viện tải file

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

# ID Google Drive chính xác của bạn
DRIVE_FILE_ID = "1aM4iLTqMu4Qd7CpYM42_oCnSbPQyFh4W"

@st.cache_resource
def get_model():
    # Tạo thư mục models nếu chưa có
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Nếu file chưa tải về hoặc là file ảo của Git LFS (<1MB), thì tải bản xịn từ Drive
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        st.info("Đang tải trọng số AI từ Google Drive xuống máy chủ, vui lòng đợi vài giây...")
        url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Tải mô hình thành công!")

    # Tiến hành load model
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