import av
import cv2
import numpy as np
import streamlit as st
import traceback
import threading
import time

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from streamlit_webrtc import (
    webrtc_streamer,
    RTCConfiguration,
    WebRtcMode,
)

from services.predictor import draw_results, get_model
from services.preprocess import preprocess_face
from detector.face_detector import detect_faces, load_cascade
from utils.style_loader import load_css

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

# ==========================================
# LUỒNG AI CHẠY NGẦM
# ==========================================
def ai_worker(state, model):
    while True:
        rois_to_process = None
        
        with state["lock"]:
            if state["latest_rois"] is not None:
                rois_to_process = state["latest_rois"]
                state["latest_rois"] = None 
                
        if rois_to_process is None:
            time.sleep(0.01)
            continue
            
        try:
            new_emotions = []
            for roi in rois_to_process:
                tensor = preprocess_face(roi)
                preds = model(tensor, training=False).numpy()[0]
                new_emotions.append(EMOTION_LABELS[int(np.argmax(preds))])
            
            with state["lock"]:
                state["latest_emotions"] = new_emotions
                
        except Exception as e:
            print("Lỗi luồng AI:", e)


# ==========================================
# KHỞI TẠO HỆ THỐNG VÀ CACHE
# ==========================================
@st.cache_resource
def init_backend_system():
    model = get_model()
    load_cascade()
    
    dummy = np.zeros((1, 48, 48, 1), dtype=np.float32)
    model(dummy, training=False)
    
    state = {
        "latest_rois": None,
        "latest_emotions": [],
        "lock": threading.Lock()
    }
    
    ai_thread = threading.Thread(target=ai_worker, args=(state, model), daemon=True)
    add_script_run_ctx(ai_thread)
    ai_thread.start()
    
    return state

shared_state = init_backend_system()


def main():
    load_css("assets/style.css")
    st.header("📷 Nhận diện cảm xúc Webcam")

    with st.spinner("Đang cấu hình luồng AI đồng bộ siêu mượt..."):
        _ = init_backend_system()

    st.write("Nhấn START để bật webcam.")

    # 1. Lấy context của luồng chính (Main Thread)
    ctx = get_script_run_ctx()

    # ==========================================
    # LUỒNG VIDEO CHÍNH (Đã đưa vào trong main)
    # ==========================================
    def video_frame_callback(frame: av.VideoFrame):
        # 2. Truyền context của luồng chính cho luồng WebRTC để tắt lỗi cảnh báo đỏ
        if ctx:
            add_script_run_ctx(threading.current_thread(), ctx)
        
        try:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # KHẮC PHỤC LỖI TÌM MẶT Ở XA: Thu nhỏ 1/2 để ảnh không bị rỗ quá mức
            small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = detect_faces(small_gray)
            
            img_h, img_w = gray.shape
            current_rois = []
            valid_boxes = []

            for (x, y, w, h) in faces:
                # Đổi hệ số nhân về 2 tương ứng với tỷ lệ 1/2
                x_real, y_real, w_real, h_real = int(x * 2), int(y * 2), int(w * 2), int(h * 2)

                x1 = int(max(0, x_real))
                y1 = int(max(0, y_real))
                x2 = int(min(img_w, x_real + w_real))
                y2 = int(min(img_h, y_real + h_real))
                
                w_c = x2 - x1
                h_c = y2 - y1

                if w_c < 30 or h_c < 30:
                    continue

                roi = gray[y1:y2, x1:x2]
                if roi.size > 0:
                    current_rois.append(roi)
                    valid_boxes.append((x1, y1, w_c, h_c))

            with shared_state["lock"]:
                if len(current_rois) > 0:
                    shared_state["latest_rois"] = current_rois
                emotions = list(shared_state["latest_emotions"])

            for i, (x1, y1, w_c, h_c) in enumerate(valid_boxes):
                label = emotions[i] if i < len(emotions) else "Detecting..."
                img = draw_results(img, x1, y1, w_c, h_c, label)
                
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            traceback.print_exc()
            return frame

    webrtc_streamer(
        key="emotion",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        async_processing=True,
        media_stream_constraints={
            "video": {
                "width": 1280, 
                "height": 720,
                "frameRate": 60,
            },
            "audio": False,
        },
    )

if __name__ == "__main__":
    main()