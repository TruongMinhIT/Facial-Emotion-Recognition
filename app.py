import streamlit as st
from utils.style_loader import load_css

st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
load_css("assets/style.css")

# Sidebar
try:
    st.sidebar.image("assets/logo.png", use_container_width=True)
except Exception:
    pass

st.sidebar.title("Facial Emotion Recognition")
st.sidebar.markdown(
    """
    ### Chức năng

    📷 **Webcam**

    🖼️ **Upload Ảnh**
    """
)

# Main
st.title("😊 Facial Expression Recognition")

st.markdown(
"""
## Chào mừng!

Ứng dụng nhận diện cảm xúc khuôn mặt sử dụng mô hình **ResNet50** được huấn luyện trên tập dữ liệu **FER2013**.

### Các cảm xúc được nhận diện

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😄 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

### Chức năng

📷 **Webcam**
- Nhận diện cảm xúc realtime.
- Phát hiện nhiều khuôn mặt.
- Hiển thị nhãn trực tiếp trên video.

🖼️ **Upload Ảnh**
- Tải ảnh JPG/PNG.
- Nhận diện tất cả khuôn mặt.
- Hiển thị xác suất của từng cảm xúc.

---

👉 **Chọn chức năng ở thanh Sidebar bên trái để bắt đầu.**
"""
)

st.info(
    "Model: ResNet50 • FER2017 • Input: 48×48 GrayScale • 7 Emotion Classes"
)