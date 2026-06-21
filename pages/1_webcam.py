import av
import cv2
import numpy as np
import streamlit as st
import traceback
import threading
import time

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

# Khai báo cứng nhãn cảm xúc để luồng phụ xử lý trực tiếp không qua Streamlit Cache
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
    """Luồng xử lý AI nhận model truyền vào trực tiếp, không bị dính lỗi Streamlit Context"""
    while True:
        img_to_process = None
        
        # Kiểm tra xem có frame mới từ webcam truyền xuống không
        with state["lock"]:
            if state["latest_frame"] is not None:
                img_to_process = state["latest_frame"].copy()
                state["latest_frame"] = None  # Đánh dấu đã lấy ảnh thành công
                
        if img_to_process is None:
            time.sleep(0.01)  # Nghỉ 10ms nếu chưa có ảnh mới để tránh giải phóng CPU quá đà
            continue
            
        try:
            # 1. Thu nhỏ ảnh đi 1/2 để Haar Cascade quét mặt mượt mà không delay
            gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = detect_faces(small_gray)
            
            new_results = []

            for (x, y, w, h) in faces:
                # Trả lại tọa độ chuẩn xác trên ảnh gốc (nhân đôi)
                x_real, y_real, w_real, h_real = x * 2, y * 2, w * 2, h * 2

                roi = gray[y_real:y_real+h_real, x_real:x_real+w_real]

                if roi.size == 0:
                    continue

                # 2. Tiền xử lý khung vuông khuôn mặt
                tensor = preprocess_face(roi)

                # 3. Dự đoán siêu tốc bằng cách gọi thẳng object thay vì dùng .predict()
                preds = model(tensor, training=False).numpy()[0]
                idx = int(np.argmax(preds))
                label = EMOTION_LABELS[idx]

                new_results.append((x_real, y_real, w_real, h_real, label))
                
            # Cập nhật kết quả tính toán mới nhất cho luồng hiển thị
            with state["lock"]:
                state["latest_results"] = new_results
                
        except Exception as e:
            print("Lỗi xử lý tại luồng ngầm AI:", e)


# ==========================================
# KHỞI TẠO HỆ THỐNG VÀ CACHE BIẾN
# ==========================================
@st.cache_resource
def init_backend_system():
    """Khởi tạo mô hình và kích hoạt luồng ngầm một lần duy nhất"""
    # Load model và cascade tại luồng chính của Streamlit
    model = get_model()
    load_cascade()
    
    # Chạy thử nghiệm 1 lần (Warmup) để tránh bị khựng ở frame đầu tiên
    dummy = np.zeros((1, 48, 48, 1), dtype=np.float32)
    model(dummy, training=False)
    
    # Tạo đối tượng bộ nhớ chia sẻ giữa 2 luồng
    state = {
        "latest_frame": None,
        "latest_results": [],
        "lock": threading.Lock()
    }
    
    # Khởi chạy luồng xử lý AI chạy ngầm xuyên suốt ứng dụng
    ai_thread = threading.Thread(target=ai_worker, args=(state, model), daemon=True)
    ai_thread.start()
    
    return state

# Lấy trạng thái chia sẻ đã được đóng băng bộ nhớ
shared_state = init_backend_system()


# ==========================================
# LUỒNG XỬ LÝ VIDEO CHÍNH CỦA WEBRTC
# ==========================================
def video_frame_callback(frame: av.VideoFrame):
    try:
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Liên tục ghi đè ảnh mới nhất để luồng AI bốc đi xử lý
        with shared_state["lock"]:
            shared_state["latest_frame"] = img.copy()
            # Lấy bản sao kết quả nhận diện mới nhất hiện tại để vẽ lên hình
            results = list(shared_state["latest_results"])
            
        # 2. Luôn luôn vẽ kết quả lên màn hình (nếu AI chưa tính xong frame mới, dùng tạm frame cũ)
        for (x, y, w, h, label) in results:
            img = draw_results(img, x, y, w, h, label)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")
        
    except Exception as e:
        traceback.print_exc()
        return frame


def main():
    load_css("assets/style.css")

    st.header("📷 Nhận diện cảm xúc Webcam")

    with st.spinner("Đang cấu hình luồng AI đồng bộ siêu mượt..."):
        # Đảm bảo hệ thống backend luôn sẵn sàng
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
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },
    )

main()