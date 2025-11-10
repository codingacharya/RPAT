import cv2
import numpy as np
import os

# ======================================================
# ⚙️ Configuration — must match train_model.py
# ======================================================
IMG_SIZE = 224
SEQUENCE_LENGTH = 20

# ======================================================
# 🎞️ Frame extraction utility
# ======================================================
def extract_frames(video_path, max_frames=SEQUENCE_LENGTH):
    """
    Extracts evenly spaced frames from a video file.
    Ensures the total number of frames equals SEQUENCE_LENGTH.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_gap = max(1, total_frames // max_frames)

    for i in range(0, total_frames, frame_gap):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to match CNN input
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0  # Normalize to [0,1]
        frames.append(frame)

        if len(frames) == max_frames:
            break

    cap.release()

    # Pad short videos with black frames
    while len(frames) < max_frames:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    return np.array(frames, dtype=np.float32)

# ======================================================
# 📦 Preprocess uploaded video (for Streamlit app)
# ======================================================
def preprocess_video(video_path):
    """
    Loads a video, extracts frames, normalizes, and returns a numpy array
    shaped as (1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames = extract_frames(video_path)
    # Add batch dimension for model prediction
    return np.expand_dims(frames, axis=0)
