import cv2
import numpy as np
from PIL import Image
import streamlit as st

from services.predictor import predict_emotion, draw_results, EMOTION_LABELS
from services.preprocess import preprocess_face
from detector.face_detector import detect_faces
from utils.style_loader import load_css


EMOTION_COLORS = {
    "Angry":    "#EF4444",
    "Disgust":  "#F97316",
    "Fear":     "#7C3AED",
    "Happy":    "#10B981",
    "Neutral":  "#64748B",
    "Sad":      "#3B82F6",
    "Surprise": "#F59E0B",
}

EMOTION_ICONS = {
    "Angry":    "😠",
    "Disgust":  "🤢",
    "Fear":     "😨",
    "Happy":    "😄",
    "Neutral":  "😐",
    "Sad":      "😢",
    "Surprise": "😲",
}

# ────────────────────────────────────────────────────────────
def run_inference(img_bgr):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    results = []
    for i, (x, y, w, h) in enumerate(faces):
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        tensor        = preprocess_face(roi)
        label, probs  = predict_emotion(tensor)
        draw_results(img_bgr, x, y, w, h, label)
        results.append({"face_id": i + 1, "emotion": label, "probs": probs})
    return img_bgr, results


def confidence_bar(label: str, value: float):
    """Render một dòng confidence bar có màu."""
    color = EMOTION_COLORS.get(label, "#6366F1")
    icon  = EMOTION_ICONS.get(label, "")
    pct   = value * 100
    # Gradient track fill
    return f"""
    <div style="margin-bottom:0.65rem;">
        <div style="display:flex; justify-content:space-between;
                    align-items:center; margin-bottom:0.3rem;">
            <span style="font-size:0.84rem; font-weight:500; color:#0F172A;
                         font-family:'DM Sans',system-ui,sans-serif;">{icon} {label}</span>
            <span style="font-size:0.75rem; font-weight:600; color:#6366F1;
                         font-family:'JetBrains Mono',monospace;">{pct:.1f}%</span>
        </div>
        <div style="height:8px; background:#EEF2FF; border-radius:999px; overflow:hidden;
                    box-shadow:inset 0 1px 3px rgba(0,0,0,0.06);">
            <div style="height:100%; width:{pct:.1f}%;
                        background:linear-gradient(90deg, {color}, {color}CC);
                        border-radius:999px;
                        box-shadow:0 1px 4px {color}50;
                        transition:width 0.7s cubic-bezier(0.4,0,0.2,1);"></div>
        </div>
    </div>
    """


# ────────────────────────────────────────────────────────────
def main():
    load_css("assets/style.css")

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sb-brand">
            <div class="sb-brand-logo">🖼️</div>
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
            <a href="/webcam" target="_self" class="sb-nav-link">
                <span class="nav-icon">📷</span>
                <span class="nav-text">Webcam Live</span>
            </a>
            <a href="/upload_image" target="_self" class="sb-nav-link active">
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
        <div class="page-hero-eyebrow">🖼️ Image Analysis</div>
        <h1>Upload Ảnh</h1>
        <p>Nhận diện biểu cảm khuôn mặt từ ảnh tĩnh · Hỗ trợ multi-face · Phân tích xác suất</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload zone ───────────────────────────────────────
    uploaded = st.file_uploader(
        "Kéo thả hoặc nhấn để chọn ảnh (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    if uploaded is None:
        st.markdown("""
        <div class="upload-empty">
            <span style="font-size:4rem; display:block; margin-bottom:1.25rem; opacity:0.25;">🖼️</span>
            <p style="color:#6B7280; font-size:0.95rem; margin:0 0 0.4rem; font-weight:500;">
                Chưa có ảnh nào được tải lên
            </p>
            <p style="color:#9CA3AF; font-size:0.83rem; margin:0;">
                Chọn ảnh <strong style="color:#6366F1;">JPG / PNG</strong> để bắt đầu nhận diện cảm xúc
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Process ───────────────────────────────────────────
    image   = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("Đang phân tích khuôn mặt..."):
        result_bgr, faces = run_inference(img_bgr)

    # No face found
    if len(faces) == 0:
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.warning("⚠️  Không phát hiện khuôn mặt nào. Thử ảnh khác với góc chụp rõ hơn.")
        return

    # ── Layout: image | results ───────────────────────────
    col_img, col_results = st.columns([1.4, 1], gap="large")

    with col_img:
        st.image(
            cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB),
            caption=f"Phát hiện {len(faces)} khuôn mặt",
            use_column_width=True,
        )

    with col_results:
        st.markdown(f"""
        <div class="result-header">
            <div class="result-header-bar"></div>
            <span class="result-header-label">Kết quả · {len(faces)} khuôn mặt</span>
        </div>
        """, unsafe_allow_html=True)

        for face in faces:
            emotion = face["emotion"]
            icon    = EMOTION_ICONS.get(emotion, "")
            color   = EMOTION_COLORS.get(emotion, "#6366F1")
            top_pct = float(np.max(face["probs"])) * 100

            # Top emotion pill + confidence bars
            bars_html = "".join(
                confidence_bar(EMOTION_LABELS[i], float(face["probs"][i]))
                for i in np.argsort(face["probs"])[::-1]
            )

            st.markdown(f"""
            <div class="face-result-card">
                <div style="display:flex; justify-content:space-between;
                            align-items:center; margin-bottom:1rem;">
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                                 color:#9CA3AF; text-transform:uppercase; letter-spacing:0.12em;
                                 font-weight:700;">Face #{face['face_id']}</span>
                    <span style="background:{color}12; border:1.5px solid {color}35;
                                 color:{color}; border-radius:999px; padding:0.3rem 0.85rem;
                                 font-size:0.82rem; font-weight:700; letter-spacing:0.01em;
                                 white-space:nowrap;">
                        {icon} {emotion} · {top_pct:.0f}%
                    </span>
                </div>
                {bars_html}
            </div>
            """, unsafe_allow_html=True)


main()