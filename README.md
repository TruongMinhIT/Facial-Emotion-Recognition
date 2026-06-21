# 😊 Facial Expression Recognition using ResNet50 + Attention

Ứng dụng nhận diện cảm xúc khuôn mặt bằng Deep Learning sử dụng **ResNet50 + Channel Attention (SE Block)** và giao diện **Streamlit**.

---

# Chức năng

- 📷 Nhận diện cảm xúc qua Webcam
- 🖼️ Nhận diện cảm xúc từ ảnh
- 😀 Nhận diện 7 loại cảm xúc

| Label | Emotion |
|-------|---------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Neutral |
| 5 | Sad |
| 6 | Surprise |

---

# Công nghệ sử dụng

- Python
- TensorFlow
- Keras
- OpenCV
- Streamlit
- Streamlit-WebRTC
- ResNet50
- Haar Cascade
- NumPy
- PIL

---

# Dataset

FER-2013

---

# Kiến trúc mô hình

```
48×48 Gray Image
        │
        ▼
Resize → 224×224
        │
        ▼
Gray → RGB
        │
        ▼
preprocess_input()
        │
        ▼
ResNet50 (ImageNet)
        │
        ▼
Channel Attention (SE Block)
        │
        ▼
Global Average Pooling
        │
        ▼
Dense
        │
        ▼
Softmax (7 classes)
```

---

# Cấu trúc thư mục

```
streamlit-facial-emotion-recognition
│
├── app.py
├── requirements.txt
├── README.md
│
├── assets
│   └── style.css
│
├── detector
│   ├── face_detector.py
│   └── haarcascade_frontalface_default.xml
│
├── models
│   └── ResNet50_Emotion_Augmentation_Attention.keras
│
├── pages
│   ├── 1_webcam.py
│   └── 2_upload_image.py
│
├── services
│   ├── predictor.py
│   └── preprocess.py
│
└── utils
    └── style_loader.py
```

---

# Yêu cầu

Khuyến nghị

- Python 3.10.x

Không khuyến nghị

- Python 3.12
- Python 3.13

vì TensorFlow 2.10 chưa hỗ trợ.

---

# Tạo môi trường ảo

## Windows

```bash
python -m venv emotion_ai
```

Kích hoạt

```bash
emotion_ai\Scripts\activate
```

---

# Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Nếu pip quá cũ:

```bash
python -m pip install --upgrade pip
```

Sau đó chạy lại

```bash
pip install -r requirements.txt
```

---

# Chạy ứng dụng

```bash
streamlit run app.py
```

Sau đó mở trình duyệt

```
http://localhost:8501
```

---

# Sử dụng Webcam

1. Chọn **Webcam**
2. Nhấn **START**
3. Cho phép trình duyệt sử dụng Camera
4. Hệ thống sẽ tự nhận diện cảm xúc realtime

---

# Sử dụng Upload Image

1. Chọn **Upload Image**
2. Chọn ảnh jpg/jpeg/png
3. Hệ thống sẽ

- Phát hiện khuôn mặt
- Nhận diện cảm xúc
- Hiển thị xác suất từng lớp

---

# Model

Model được lưu tại

```
models/
```

Tên file

```
ResNet50_Emotion_Augmentation_Attention.keras
```

---

# Giao diện

Ứng dụng gồm

- Trang chủ
- Webcam Recognition
- Image Recognition

---

# Requirements

Project được xây dựng với

- TensorFlow 2.10.1
- Keras 2.10.0
- Streamlit 1.31
- OpenCV 4.10
- Python 3.10

---
