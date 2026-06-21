import streamlit as st
from utils.style_loader import load_css

st.set_page_config(
    page_title="FaceEmotion AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css("assets/style.css")

# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-logo">🧠</div>
        <div class="sb-brand-text">
            <h2>FaceEmotion AI</h2>
            <p>v2.0 · ResNet50</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status
    st.markdown("""
    <div class="sb-status">
        <div class="sb-status-dot"></div>
        Model · Online
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-nav">
        <a href="/" target="_self" class="sb-nav-link active">
            <span class="nav-icon">🏠</span>
            <span class="nav-text">Tổng quan</span>
        </a>
        <a href="/webcam" target="_self" class="sb-nav-link">
            <span class="nav-icon">📷</span>
            <span class="nav-text">Webcam Live</span>
        </a>
        <a href="/upload_image" target="_self" class="sb-nav-link">
            <span class="nav-icon">🖼️</span>
            <span class="nav-text">Upload Ảnh</span>
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Model info
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
        <div class="sb-model-row">
            <span class="sb-model-key">Framework</span>
            <span class="sb-model-val">TensorFlow</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="sb-footer">© 2025 FaceEmotion AI · HCMUTE</div>
    """, unsafe_allow_html=True)


# ─── Hero Header ───────────────────────────────────────────
st.markdown("""
<div class="page-hero">
    <div class="page-hero-eyebrow">
        ✦ AI · Computer Vision · Deep Learning
    </div>
    <h1>Facial Expression<br>Recognition</h1>
    <p>Nhận diện biểu cảm khuôn mặt theo thời gian thực bằng ResNet50 &amp; FER2017</p>
</div>
""", unsafe_allow_html=True)

# ─── Stats ─────────────────────────────────────────────────
st.markdown("""
<div class="stat-grid">
    <div class="stat-item">
        <span class="stat-number">7</span>
        <span class="stat-label">Emotion Classes</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">48px</span>
        <span class="stat-label">Input Resolution</span>
    </div>
    <div class="stat-item">
        <span class="stat-number">RT</span>
        <span class="stat-label">Real-time Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Emotion tags ───────────────────────────────────────────
st.markdown("""
<div class="tech-card" style="margin-bottom:1.5rem;">
    <p style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; letter-spacing:0.14em;
              text-transform:uppercase; color:#9CA3AF; margin:0 0 0.85rem; font-weight:700;">
        7 Cảm xúc được nhận diện
    </p>
    <div class="emotion-grid">
        <span class="emotion-tag etag-angry">😠 Angry</span>
        <span class="emotion-tag etag-disgust">🤢 Disgust</span>
        <span class="emotion-tag etag-fear">😨 Fear</span>
        <span class="emotion-tag etag-happy">😄 Happy</span>
        <span class="emotion-tag etag-neutral">😐 Neutral</span>
        <span class="emotion-tag etag-sad">😢 Sad</span>
        <span class="emotion-tag etag-surprise">😲 Surprise</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Feature cards ─────────────────────────────────────────
st.markdown("""
<div class="feature-row">
    <div class="feature-card">
        <div class="feature-card-glow"></div>
        <span class="fc-icon">📷</span>
        <h3>Webcam Live</h3>
        <ul>
            <li>Nhận diện cảm xúc thời gian thực qua camera</li>
            <li>Phát hiện nhiều khuôn mặt cùng lúc</li>
            <li>Điều chỉnh độ phân giải &amp; frame rate</li>
            <li>Hiển thị nhãn trực tiếp trên video stream</li>
        </ul>
    </div>
    <div class="feature-card">
        <div class="feature-card-glow"></div>
        <span class="fc-icon">🖼️</span>
        <h3>Upload Ảnh</h3>
        <ul>
            <li>Tải lên ảnh JPG / PNG bất kỳ</li>
            <li>Nhận diện tất cả khuôn mặt trong ảnh</li>
            <li>Biểu đồ xác suất 7 cảm xúc chi tiết</li>
            <li>Kết quả phân tích từng khuôn mặt riêng biệt</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Model badge ───────────────────────────────────────────
st.markdown("""
<div class="model-badge">
    <span class="dot"></span>
    ResNet50 · Attention · FER2017 Augmented · TF/Keras
</div>
""", unsafe_allow_html=True)