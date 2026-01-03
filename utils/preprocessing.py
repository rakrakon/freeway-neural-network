import numpy as np

from config.config import TrainConfig
import cv2

def preprocess_frame(frame, train_config: TrainConfig):
    """Preprocess Atari frame: grayscale, resize, normalize"""
    # Convert to grayscale
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])

    # Resize to configured dimensions (84x84)
    resized = cv2.resize(
        gray,
        (train_config.frame_width, train_config.frame_height),
        interpolation=cv2.INTER_AREA
    )

    # Keep as uint8 to save memory (normalized in model forward)
    return resized.astype(np.uint8)