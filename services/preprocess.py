import cv2
import numpy as np

IMG_SIZE = (48, 48)


def preprocess_face(face_roi_gray: np.ndarray) -> np.ndarray:
    """
    Input:
        ROI grayscale

    Output:
        Tensor (1,48,48,1)
    """

    face = cv2.resize(
        face_roi_gray,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    # KHÔNG CHIA 255
    face = face.astype(np.float32)

    face = np.expand_dims(face, axis=-1)

    face = np.expand_dims(face, axis=0)

    return face