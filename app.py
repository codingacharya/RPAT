import streamlit as st
import os
import logging
from utils.predict import predict_violence
from utils.visualization import show_video_preview, show_class_probabilities, show_sample_frames

# ============================
# CONFIGURE LOGGING
# ============================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================
# STREAMLIT APP SETUP
# ============================
st.set_page_config(page_title="Violence Detection App", page_icon="🎥", layout="wide")

st.title("🔍 Violence Detection from Video")
st.write("Upload a short video clip to analyze whether it contains **violent** or **non-violent** scenes.")

# Ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)

# ============================
# VIDEO UPLOAD SECTION
# ============================
uploaded_file = st.file_uploader("📁 Upload your video file (MP4 preferred):", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to local directory
    video_path = os.path.join("uploads", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    logging.info(f"User uploaded: {uploaded_file.name}")

    # Display video preview
    show_video_preview(video_path)

    # Optional: show sample frames
    with st.expander("🖼️ Preview extracted frames"):
        show_sample_frames(video_path)

    # ============================
    # RUN PREDICTION
    # ============================
    if st.button("🚀 Analyze Video"):
        try:
            st.info("⏳ Processing video and running model prediction...")
            result = predict_violence(video_path)

            label = result["label"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]

            # Log prediction
            logging.info(f"Prediction - File: {uploaded_file.name}, Label: {label}, Confidence: {confidence:.2f}")

            # Display result
            st.success(f"**Prediction:** {label} ({confidence*100:.2f}% confidence)")
            show_class_probabilities(probabilities)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            logging.error("Error during prediction", exc_info=True)

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "Developed with ❤️ using Streamlit + TensorFlow<br>"
    "<b>Violence Detection Model (MobileNetV2 + BiLSTM)</b>"
    "</div>",
    unsafe_allow_html=True
)
