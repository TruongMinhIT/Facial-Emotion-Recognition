import cv2
import numpy as np
import streamlit as st

CASCADE_PATH = "detector/haarcascade_frontalface_default.xml"


@st.cache_resource
def load_cascade():
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise FileNotFoundError(
            f"Không tìm thấy file cascade tại: {CASCADE_PATH}\n"
            "Tải về tại: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml"
        )
    return cascade


def detect_faces(gray_frame: np.ndarray) -> list:
    """
    Nhận frame grayscale, trả về list (x, y, w, h).
    Tự load cascade bên trong — caller không cần truyền cascade.
    """
    cascade = load_cascade()
    faces = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return list(faces) if len(faces) > 0 else []