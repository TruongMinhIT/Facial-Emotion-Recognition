import av
import cv2
import numpy as np
import streamlit as st
import traceback
import threading
import time

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

from services.predictor import draw_results, get_model
from services.preprocess import preprocess_face
from detector.face_detector import detect_faces, load_cascade
from utils.style_loader import load_css

# ─── Config ────────────────────────────────────────────────
# FIX LỖI 1: Tắt STUN Server để webcam bật lên ngay lập tức trong 1 giây
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": []
})

EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprise",
}

RESOLUTION_OPTIONS = {
    "HD 720p (1280×720) — Khuyến nghị": (1280, 720),
    "Full HD 1080p (1920×1080)":         (1920, 1080),
    "Standard 480p (854×480)":           (854, 480),
    "Compact 360p (640×360) — Tiết kiệm": (640, 360),
}

SCALE_OPTIONS = {
    "1/2 — Cân bằng tốt nhất": 0.5,
    "1/3 — Nhanh hơn":          1/3,
    "1/4 — Rất nhanh":          0.25,
}

# ─── AI Background Worker ───────────────────────────────────
def ai_worker(state, model):
    while True:
        rois_to_process = None
        with state["lock"]:
            if state["latest_rois"] is not None:
                rois_to_process = state["latest_rois"]
                state["latest_rois"] = None
                
        # Bỏ qua nếu không có khuôn mặt nào được gửi đến
        if rois_to_process is None or len(rois_to_process) == 0:
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
            print("AI worker error:", e)


@st.cache_resource
def init_backend():
    model = get_model()
    load_cascade()
    dummy = np.zeros((1, 48, 48, 1), dtype=np.float32)
    model(dummy, training=False)
    state = {
        "latest_rois": None,
        "latest_emotions": [],
        "lock": threading.Lock(),
    }
    t = threading.Thread(target=ai_worker, args=(state, model), daemon=True)
    add_script_run_ctx(t)
    t.start()
    return state


# ─── Page ──────────────────────────────────────────────────
def main():
    load_css("assets/style.css")

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sb-brand">
            <div class="sb-brand-logo">📷</div>
            <div class="sb-brand-text">
                <h2>FaceEmotion AI</h2>
                <p>v2.0 · ResNet50</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sb-status">
            <div class="sb-status-dot"></div>
            Model · Online
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sb-nav">
            <a href="/" target="_self" class="sb-nav-link">
                <span class="nav-icon">🏠</span>
                <span class="nav-text">Tổng quan</span>
            </a>
            <a href="/webcam" target="_self" class="sb-nav-link active">
                <span class="nav-icon">📷</span>
                <span class="nav-text">Webcam Live</span>
            </a>
            <a href="/upload_image" target="_self" class="sb-nav-link">
                <span class="nav-icon">🖼️</span>
                <span class="nav-text">Upload Ảnh</span>
            </a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Thông tin mô hình</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sb-model-card">
            <div class="sb-model-row">
                <span class="sb-model-key">Architecture</span>
                <span class="sb-model-val">ResNet50</span>
            </div>
            <div class="sb-model-row">
                <span class="sb-model-key">Dataset</span>
                <span class="sb-model-val">FER2017</span>
            </div>
            <div class="sb-model-row">
                <span class="sb-model-key">Input</span>
                <span class="sb-model-val">48×48 Gray</span>
            </div>
            <div class="sb-model-row">
                <span class="sb-model-key">Classes</span>
                <span class="sb-model-val">7 emotions</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-footer">© 2025 FaceEmotion AI · HCMUTE</div>', unsafe_allow_html=True)

    # ── Page Header ───────────────────────────────────────
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-eyebrow">📷 Real-time Detection</div>
        <h1>Webcam Live</h1>
        <p>Nhận diện biểu cảm khuôn mặt theo thời gian thực · Hỗ trợ multi-face</p>
    </div>
    """, unsafe_allow_html=True)

    shared_state = init_backend()
    ctx = get_script_run_ctx()

    # ── Camera Settings ────────────────────────────────────
    with st.expander("⚙️  Cài đặt Camera & Xử lý", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="camera-selector-title">📹 Độ phân giải</div>', unsafe_allow_html=True)
            res_label = st.selectbox(
                "Độ phân giải", options=list(RESOLUTION_OPTIONS.keys()),
                index=0, label_visibility="collapsed",
            )
            res_w, res_h = RESOLUTION_OPTIONS[res_label]

        with col2:
            st.markdown('<div class="camera-selector-title">🎞️ Frame Rate</div>', unsafe_allow_html=True)
            fps = st.select_slider(
                "FPS", options=[15, 24, 30, 60], value=30,
                label_visibility="collapsed",
            )

        with col3:
            st.markdown('<div class="camera-selector-title">⚡ Scale xử lý AI</div>', unsafe_allow_html=True)
            scale_label = st.selectbox(
                "Scale", options=list(SCALE_OPTIONS.keys()),
                index=0, label_visibility="collapsed",
            )
            proc_scale = SCALE_OPTIONS[scale_label]

    # ── Live Banner ───────────────────────────────────────
    st.markdown("""
    <div class="live-banner">
        <div class="live-dot"></div>
        <span class="live-text">LIVE · Nhấn START để bật camera</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Video callback ────────────────────────────────────
    def video_frame_callback(frame: av.VideoFrame):
        if ctx:
            add_script_run_ctx(threading.current_thread(), ctx)
        try:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Logic thu nhỏ của bạn rất chuẩn xác
            small_gray = cv2.resize(gray, (0, 0), fx=proc_scale, fy=proc_scale)
            faces = detect_faces(small_gray)
            
            inv = 1.0 / proc_scale
            img_h, img_w = gray.shape
            current_rois, valid_boxes = [], []
            
            for (x, y, w, h) in faces:
                x1 = int(max(0, x * inv))
                y1 = int(max(0, y * inv))
                x2 = int(min(img_w, (x + w) * inv))
                y2 = int(min(img_h, (y + h) * inv))
                wc, hc = x2 - x1, y2 - y1
                
                if wc < 30 or hc < 30:
                    continue
                    
                roi = gray[y1:y2, x1:x2]
                if roi.size > 0:
                    current_rois.append(roi)
                    valid_boxes.append((x1, y1, wc, hc))
                    
            with shared_state["lock"]:
                if len(current_rois) > 0:
                    shared_state["latest_rois"] = current_rois
                else:
                    # FIX LỖI 2: Xóa bộ đệm AI ngay lập tức khi khuôn mặt rời khỏi khung hình
                    # Ngăn chặn việc hiển thị kết quả ảo bị kẹt từ hàng ngàn frame trước
                    shared_state["latest_emotions"] = []
                    
                emotions = list(shared_state["latest_emotions"])
                
            for i, (x1, y1, wc, hc) in enumerate(valid_boxes):
                label = emotions[i] if i < len(emotions) else "Detecting..."
                img = draw_results(img, x1, y1, wc, hc, label)
                
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception:
            traceback.print_exc()
            return frame

    webrtc_streamer(
        key="emotion", # FIX LỖI 3: Để key cố định, KHÔNG gán biến độ phân giải vào đây
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        async_processing=True,
        media_stream_constraints={
            "video": {
                "width":     {"ideal": res_w},
                "height":    {"ideal": res_h},
                "frameRate": {"ideal": fps},
            },
            "audio": False,
        },
    )

    # ── Tips ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="tips-card">
        <div class="tips-card-title">💡 Mẹo sử dụng</div>
        <ul style="margin:0; padding-left:0; list-style:none;">
            <li style="color:#374151; font-size:0.85rem; margin-bottom:0.45rem; display:flex; gap:0.6rem; align-items:flex-start;">
                <span style="color:#6366F1; font-weight:700; flex-shrink:0;">→</span>
                Đảm bảo đủ ánh sáng — khuôn mặt chiếu sáng từ phía trước
            </li>
            <li style="color:#374151; font-size:0.85rem; margin-bottom:0.45rem; display:flex; gap:0.6rem; align-items:flex-start;">
                <span style="color:#6366F1; font-weight:700; flex-shrink:0;">→</span>
                Giữ khoảng cách 40–80 cm với camera để nhận diện tốt nhất
            </li>
            <li style="color:#374151; font-size:0.85rem; margin-bottom:0.45rem; display:flex; gap:0.6rem; align-items:flex-start;">
                <span style="color:#6366F1; font-weight:700; flex-shrink:0;">→</span>
                Nếu bị giật, giảm Frame Rate xuống 15–24 FPS hoặc dùng Scale 1/3
            </li>
            <li style="color:#374151; font-size:0.85rem; display:flex; gap:0.6rem; align-items:flex-start;">
                <span style="color:#6366F1; font-weight:700; flex-shrink:0;">→</span>
                Máy có nhiều camera: thử đổi thiết bị camera ở cài đặt phía trên
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()