import av
import cv2
import numpy as np
import streamlit as st
import traceback
import threading
import time

# Thêm thư viện để cấp phát Context cho luồng ngầm của Streamlit
from streamlit.runtime.scriptrunner import add_script_run_ctx

from streamlit_webrtc import (
    webrtc_streamer,
    RTCConfiguration,
    WebRtcMode,
)

# Chỉ import các hàm vẽ và xử lý phụ trợ để tránh xung đột luồng
from services.predictor import draw_results, get_model
from services.preprocess import preprocess_face
from detector.face_detector import detect_faces, load_cascade
from utils.style_loader import load_css

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# Khai báo cứng nhãn cảm xúc
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
# LUỒNG AI CHẠY NGẦM HOÀN TOÀN ĐỘC LẬP
# ==========================================
def ai_worker(state, model):
    """Luồng xử lý AI nhận model truyền vào trực tiếp"""
    while True:
        img_to_process = None
        
        with state["lock"]:
            if state["latest_frame"] is not None:
                img_to_process = state["latest_frame"].copy()
                state["latest_frame"] = None 
                
        if img_to_process is None:
            time.sleep(0.01)
            continue
            
        try:
            # Thu nhỏ 1/4 để bù trừ cho khung hình 720p to lớn
            gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
            faces = detect_faces(small_gray)
            
            img_h, img_w = gray.shape
            new_results = []

            for (x, y, w, h) in faces:
                # Ép kiểu int() và nhân 4 để lấy tọa độ thực tế trên ảnh 720p
                x_real, y_real, w_real, h_real = int(x * 4), int(y * 4), int(w * 4), int(h * 4)

                # CHẶN VIỀN: Không cho tọa độ bị âm hoặc vượt qua mép ảnh
                x1 = int(max(0, x_real))
                y1 = int(max(0, y_real))
                x2 = int(min(img_w, x_real + w_real))
                y2 = int(min(img_h, y_real + h_real))
                
                w_clamped = x2 - x1
                h_clamped = y2 - y1

                # CHỐNG CRASH: Loại bỏ các vệt bóng mờ ảo (kích thước quá nhỏ) khi di chuyển nhanh
                if w_clamped < 30 or h_clamped < 30:
                    continue

                roi = gray[y1:y2, x1:x2]

                if roi.size == 0:
                    continue

                tensor = preprocess_face(roi)
                preds = model(tensor, training=False).numpy()[0]
                idx = int(np.argmax(preds))
                label = EMOTION_LABELS[idx]

                # LƯU KẾT QUẢ ĐỂ VẼ: Gửi bộ tọa độ đã an toàn tuyệt đối ra ngoài
                new_results.append((x1, y1, w_clamped, h_clamped, label))
                
            with state["lock"]:
                state["latest_results"] = new_results
                
        except Exception as e:
            print("Lỗi xử lý tại luồng ngầm AI:", e)


# ==========================================
# KHỞI TẠO HỆ THỐNG VÀ CACHE BIẾN
# ==========================================
@st.cache_resource
def init_backend_system():
    model = get_model()
    load_cascade()
    
    dummy = np.zeros((1, 48, 48, 1), dtype=np.float32)
    model(dummy, training=False)
    
    state = {
        "latest_frame": None,
        "latest_results": [],
        "lock": threading.Lock()
    }
    
    ai_thread = threading.Thread(target=ai_worker, args=(state, model), daemon=True)
    add_script_run_ctx(ai_thread)
    ai_thread.start()
    
    return state

shared_state = init_backend_system()


# ==========================================
# LUỒNG XỬ LÝ VIDEO CHÍNH CỦA WEBRTC
# ==========================================
def video_frame_callback(frame: av.VideoFrame):
    # CẤP QUYỀN CHO LUỒNG WEBRTC: Triệt tiêu dòng chữ đỏ "missing ScriptRunContext"
    add_script_run_ctx(threading.current_thread())
    
    try:
        img = frame.to_ndarray(format="bgr24")
        
        with shared_state["lock"]:
            shared_state["latest_frame"] = img.copy()
            results = list(shared_state["latest_results"])
            
        for (x, y, w, h, label) in results:
            img = draw_results(img, x, y, w, h, label)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")
        
    except Exception as e:
        traceback.print_exc()
        return frame


def main():
    load_css("assets/style.css")

    st.header("📷 Nhận diện cảm xúc Webcam")

    with st.spinner("Đang khởi động AI đa luồng..."):
        _ = init_backend_system()

    st.write("Nhấn START để bật webcam.")

    webrtc_streamer(
        key="emotion",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        async_processing=True,
        media_stream_constraints={
            "video": {
                # Ép buộc luồng video chạy đúng chuẩn HD 720p, không để trình duyệt tự làm mờ
                "width": {"min": 1280, "ideal": 1280},
                "height": {"min": 720, "ideal": 720},
                "frameRate": {"ideal": 60},
            },
            "audio": False,
        },
    )

if __name__ == "__main__":
    main()