# 🧠 FaceEmotion AI - Hệ Thống Nhận Diện Cảm Xúc Thời Gian Thực

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![WebRTC](https://img.shields.io/badge/WebRTC-Realtime-333333?logo=webrtc&logoColor=white)

---

## 🎓 Nhóm Thực Hiện (Nhóm 8)
* **Nguyễn Trường Minh** - 23110125
* **Đoàn Quốc Huy** - 23110103
* **Lương Quang Lâm** - 23110121
* **Đoàn Ngọc Mạnh** - 23110124
---


**FaceEmotion AI** là ứng dụng Web Application được xây dựng để nhận diện và phân loại 7 trạng thái biểu cảm khuôn mặt của con người (Tức giận, Ghê tởm, Sợ hãi, Hạnh phúc, Bình thường, Buồn bã, Ngạc nhiên). Hệ thống vận hành dựa trên kiến trúc Mạng nơ-ron tích chập **ResNet50** (kết hợp cơ chế Attention) và giao diện trực quan từ **Streamlit**.

Dự án được thiết kế đặc biệt để giải quyết bài toán hiệu năng (Real-time Inference) trên trình duyệt bằng cách áp dụng kiến trúc **Đa luồng Bất đồng bộ (Asynchronous Multithreading)**, giúp xử lý luồng video chuẩn HD cực kỳ mượt mà với độ trễ tiệm cận 0ms.

---

## ✨ Tính Năng Nổi Bật

### 📷 1. Webcam Live (Nhận Diện Thời Gian Thực)
* **Asynchronous Tracking:** Tách biệt hoàn toàn luồng quét Video và luồng dự đoán AI ngầm. Khung vẽ nhận diện (Bounding box) bám chặt vào khuôn mặt mà không bị khựng (lag) chờ AI.
* **Chống Crash Tuyệt Đối (Negative Index Clamp):** Tích hợp thuật toán chặn viền ảnh và bộ lọc nhiễu chuyển động, đảm bảo ứng dụng không bao giờ bị sập khi người dùng di chuyển nhanh ra khỏi rìa camera.
* **Giao diện Điều khiển Trực tiếp (Live Controls):** Người dùng có thể tùy chỉnh luồng Video ngay trên UI:
  * 📹 **Độ phân giải:** Hỗ trợ từ 360p tiết kiệm đến chuẩn HD 720p hoặc Full HD 1080p.
  * 🎞️ **Frame Rate:** Tùy chỉnh FPS (15, 24, 30, 60) để phù hợp với phần cứng.
  * ⚡ **Scale xử lý AI:** Tùy chọn thu nhỏ khung hình nội suy (1/2, 1/3, 1/4) để tối ưu tốc độ tính toán cho các máy tính cấu hình thấp.
* **Multi-face Tracking:** Hỗ trợ nhận diện độc lập nhiều khuôn mặt trong cùng một khung hình.

### 🖼️ 2. Upload Ảnh (Phân Tích Ảnh Tĩnh)
* Tải lên các định dạng phổ biến (`.jpg`, `.jpeg`, `.png`).
* Nhận diện toàn bộ khuôn mặt trong ảnh đám đông.
* Trích xuất kết quả dự đoán chi tiết kèm **Biểu đồ xác suất** trực quan cho từng cảm xúc.

---

## 🛠️ Công Nghệ Sử Dụng

* **Ngôn Ngữ Lập Trình:** Python
* **Frontend / UI:** Streamlit, Custom HTML/CSS
* **Truyền Phát Video (P2P):** Streamlit-WebRTC
* **Computer Vision:** OpenCV (Haar Cascade Classifier), NumPy
* **Deep Learning Framework:** TensorFlow / Keras

---

## 🚀 Hướng Dẫn Cài Đặt (Chạy Local)

### 1. Yêu cầu hệ thống
* Trình duyệt web (khuyến nghị Google Chrome hoặc Edge).
* Đã cài đặt **Python 3.9** trở lên.
* Thiết bị có hỗ trợ Webcam.

### 2. Các bước triển khai

**Bước 1:** Clone repository về máy tính:
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

**Bước 2:** (Khuyến nghị) Tạo và kích hoạt môi trường ảo (Virtual Environment):
```bash
# Đối với Windows
python -m venv venv
venv\Scripts\activate

# Đối với MacOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**Bước 3:** Cài đặt các gói thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```
*(Lưu ý: File `requirements.txt` cần bao gồm `streamlit`, `streamlit-webrtc`, `opencv-python-headless`, `tensorflow`, `numpy`, `Pillow`, `av`)*

**Bước 4:** Khởi chạy Server Streamlit:
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 📂 Cấu Trúc Mã Nguồn

```text
📁 project-root/
│
├── 📄 app.py                  # Điểm neo khởi chạy, cấu hình UI/UX và Sidebar
├── 📁 pages/
│   ├── 📄 1_webcam.py         # Module Asynchronous Tracking cho luồng Webcam
│   └── 📄 2_upload_image.py   # Module phân tích ảnh tĩnh
│
├── 📁 models/
│   └── 📄 ResNet50_Emotion_Augmentation_Attention.keras # Trọng số mô hình đã huấn luyện
│
├── 📁 detector/
│   └── 📄 face_detector.py    # Logic nhận diện tọa độ khuôn mặt (Haar Cascade)
│
├── 📁 services/
│   ├── 📄 predictor.py        # Object Detection, load mô hình và hàm dự đoán
│   └── 📄 preprocess.py       # Tiền xử lý tensor (resize, grayscale) trước khi đưa vào mô hình
│
├── 📁 utils/
│   └── 📄 style_loader.py     # Hàm inject file CSS tĩnh vào Streamlit
│
└── 📁 assets/
    └── 📄 style.css           # Cấu trúc giao diện, Dark mode, Styling elements
```

---

## 📝 Lưu Ý Triển Khai Đám Mây (Deployment)
Phiên bản chạy Local đã được cấu hình tối ưu để loại bỏ máy chủ STUN (`"iceServers": []`), giúp khởi động camera ngay lập tức. Tuy nhiên, nếu bạn muốn đưa dự án này lên **Streamlit Cloud, Heroku, hoặc Render**, bạn **bắt buộc** phải cấu hình lại máy chủ STUN của Google trong `1_webcam.py` để WebRTC có thể định tuyến NAT:

```python
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
})
```
