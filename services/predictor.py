import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown  

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

DRIVE_FILE_ID = "1aM4iLTqMu4Qd7CpYM42_oCnSbPQyFh4W"

@st.cache_resource
def get_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Hàm tải file chuẩn xác nhất với gdown mới
    def force_download():
        st.warning("⚠️ Đang tải AI (164MB) từ mây về máy chủ. Lần đầu sẽ hơi lâu, bạn đừng chuyển trang nhé...")
        if os.path.exists(MODEL_PATH):
            try:
                os.remove(MODEL_PATH)
            except:
                pass
        # Chỉ dùng id= và output=, bỏ qua url và fuzzy
        gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
        st.success("Tải mô hình thành công! Đang khởi động AI...")

    # KIỂM TRA LỚP 1: Chắc chắn file phải nặng hơn 100MB (100,000,000 bytes)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000000:
        force_download()

    # KIỂM TRA LỚP 2: Bẫy lỗi. Nếu file đủ dung lượng nhưng bị rách/hỏng thì ép tải lại
    try:
        model = load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"preprocess_input": preprocess_input}
        )
    except Exception as e:
        force_download()
        model = load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"preprocess_input": preprocess_input}
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