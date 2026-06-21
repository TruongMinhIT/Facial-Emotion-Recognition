import cv2
import numpy as np
from PIL import Image
import streamlit as st

from services.predictor import (
    predict_emotion,
    draw_results,
    EMOTION_LABELS,
)

from services.preprocess import preprocess_face
from detector.face_detector import detect_faces
from utils.style_loader import load_css


def run_inference(img_bgr):

    gray = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    faces = detect_faces(gray)

    results = []

    for i, (x, y, w, h) in enumerate(faces):

        roi = gray[y:y+h, x:x+w]

        if roi.size == 0:
            continue

        tensor = preprocess_face(roi)

        label, probs = predict_emotion(tensor)

        draw_results(
            img_bgr,
            x,
            y,
            w,
            h,
            label,
        )

        results.append(
            {
                "face_id": i + 1,
                "emotion": label,
                "probs": probs,
            }
        )

    return img_bgr, results


def main():

    load_css("assets/style.css")

    st.header("🖼️ Nhận diện cảm xúc từ ảnh")

    uploaded = st.file_uploader(
        "Chọn ảnh",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")

    img_rgb = np.array(image)

    img_bgr = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2BGR,
    )

    result, faces = run_inference(img_bgr)

    st.image(
        cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
        use_column_width=True,
    )

    if len(faces) == 0:
        st.warning("Không phát hiện khuôn mặt.")
        return

    for face in faces:

        st.subheader(
            f"Khuôn mặt {face['face_id']}"
        )

        st.success(face["emotion"])

        labels = list(EMOTION_LABELS.values())

        st.bar_chart(
            dict(
                zip(
                    labels,
                    face["probs"],
                )
            )
        )


main()