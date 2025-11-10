import tensorflow as tf
import joblib
import numpy as np
import os
from utils.preprocess import preprocess_video

# ============================
# CONFIG
# ============================
MODEL_PATH = "model/violence_model.h5"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"

# ============================
# LOAD MODEL & ENCODER
# ============================
@tf.keras.utils.register_keras_serializable()
def load_model_and_encoder():
    """
    Loads the trained Keras model and the label encoder.
    Uses Streamlit or manual caching to avoid repeated loading.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"❌ Label encoder not found at {LABEL_ENCODER_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, label_encoder

# ============================
# PREDICTION FUNCTION
# ============================
def predict_violence(video_path):
    """
    Runs prediction on a given video file.
    Returns the predicted label and confidence score.
    """
    # Load model + encoder
    model, label_encoder = load_model_and_encoder()

    # Preprocess video
    video_data = preprocess_video(video_path)

    # Predict probabilities
    preds = model.predict(video_data)
    confidence = np.max(preds)
    pred_class = np.argmax(preds)

    # Decode label
    label = label_encoder.inverse_transform([pred_class])[0]

    result = {
        "label": label,
        "confidence": float(confidence),
        "probabilities": {label_encoder.classes_[i]: float(preds[0][i]) for i in range(len(label_encoder.classes_))}
    }

    return result

# ============================
# TESTING (Run standalone)
# ============================
if __name__ == "__main__":
    test_video = "uploads/sample_video.mp4"
    if os.path.exists(test_video):
        res = predict_violence(test_video)
        print("✅ Prediction Result:")
        print(res)
    else:
        print("⚠️ Please place a test video at 'uploads/sample_video.mp4'")
