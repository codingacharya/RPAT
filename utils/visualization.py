import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tempfile

# ============================
# DISPLAY CLASS PROBABILITIES
# ============================
def show_class_probabilities(probabilities: dict):
    """
    Displays the prediction probabilities using Streamlit progress bars and a chart.
    Args:
        probabilities (dict): {label: probability}
    """
    st.subheader("🎯 Prediction Confidence")
    for label, prob in probabilities.items():
        st.write(f"**{label}:** {prob * 100:.2f}%")
        st.progress(float(prob))

    # Plot using matplotlib for better visuals
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = list(probabilities.keys())
    values = [probabilities[l] for l in labels]

    ax.barh(labels, values)
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)
    ax.set_title("Class Probabilities")
    st.pyplot(fig)

# ============================
# SHOW VIDEO PREVIEW
# ============================
def show_video_preview(video_file):
    """
    Displays the uploaded video directly in Streamlit.
    """
    st.subheader("🎥 Uploaded Video Preview")
    st.video(video_file)

# ============================
# SHOW SAMPLE FRAMES (Optional)
# ============================
def show_sample_frames(video_path, num_frames=5):
    """
    Extracts and displays sample frames from the video to give a visual preview
    of what’s being analyzed.
    """
    st.subheader("🖼️ Sample Frames Extracted")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total - 1, num_frames).astype(int)
    cols = st.columns(num_frames)

    for i, frame_idx in enumerate(frame_idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp_file.name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cols[i].image(frame, use_container_width=True)

    cap.release()

# ============================
# EXAMPLE USAGE (Run standalone)
# ============================
if __name__ == "__main__":
    st.title("🔍 Visualization Test")

    sample_probs = {"NonViolence": 0.22, "Violence": 0.78}
    show_class_probabilities(sample_probs)
