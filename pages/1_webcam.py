import av
import cv2
import numpy as np
import streamlit as st
import traceback

from streamlit_webrtc import (
    webrtc_streamer,
    RTCConfiguration,
    WebRtcMode,
)

from services.predictor import (
    predict_emotion,
    draw_results,
    get_model,
)

from services.preprocess import preprocess_face
from detector.face_detector import detect_faces, load_cascade
from utils.style_loader import load_css


RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
    }
)


@st.cache_resource
def warmup_ai():

    model = get_model()

    load_cascade()

    dummy = np.zeros(
        (1, 48, 48, 1),
        dtype=np.float32,
    )

    model(dummy, training=False)

    return True


# def video_frame_callback(frame: av.VideoFrame):

#     try:

#         img = frame.to_ndarray(format="bgr24")

#         gray = cv2.cvtColor(
#             img,
#             cv2.COLOR_BGR2GRAY,
#         )

#         faces = detect_faces(gray)

#         for (x, y, w, h) in faces:

#             roi = gray[y:y+h, x:x+w]

#             if roi.size == 0:
#                 continue

#             tensor = preprocess_face(roi)

#             label, _ = predict_emotion(tensor)

#             img = draw_results(
#                 img,
#                 x,
#                 y,
#                 w,
#                 h,
#                 label,
#             )

#         return av.VideoFrame.from_ndarray(
#             img,
#             format="bgr24",
#         )

#     except Exception:

#         traceback.print_exc()

#         return frame


# def video_frame_callback(frame):
#     print(">>>> CALLBACK RUNNING <<<<")

#     img = frame.to_ndarray(format="bgr24")

#     return av.VideoFrame.from_ndarray(img, format="bgr24")

def video_frame_callback(frame: av.VideoFrame):

    try:
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detect_faces(gray)

        print("Faces:", len(faces))

        for (x, y, w, h) in faces:

            roi = gray[y:y+h, x:x+w]

            if roi.size == 0:
                continue

            print("Preprocess...")

            tensor = preprocess_face(roi)

            print("Predict...")

            label, probs = predict_emotion(tensor)

            print("Emotion:", label)

            img = draw_results(
                img,
                x,
                y,
                w,
                h,
                label,
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24",
        )

    except Exception as e:
        print("=" * 50)
        print(e)
        traceback.print_exc()
        print("=" * 50)

        return frame


def main():

    load_css("assets/style.css")

    st.header("📷 Nhận diện cảm xúc Webcam")

    with st.spinner("Đang khởi động AI..."):

        warmup_ai()

    st.write("Nhấn START để bật webcam.")

    webrtc_streamer(

        key="emotion",

        mode=WebRtcMode.SENDRECV,

        rtc_configuration=RTC_CONFIGURATION,

        video_frame_callback=video_frame_callback,

        async_processing=True,

        media_stream_constraints={
            "video": {
                "width": 320,
                "height": 240,
                "frameRate": 15,
            },
            "audio": False,
        },
    )


main()