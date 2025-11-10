import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, GlobalAveragePooling2D, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import joblib

# ✅ Import preprocessing functions
from utils.preprocess import extract_frames

# ======================================================
# ⚙️ Configuration
# ======================================================
DATASET_DIR = "dataset"  # dataset folder containing Violence/ and NonViolence/
IMG_SIZE = 224
SEQUENCE_LENGTH = 20
BATCH_SIZE = 2
EPOCHS = 15

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "violence_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# ======================================================
# 📂 Load dataset
# ======================================================
X, y = [], []
for label in os.listdir(DATASET_DIR):
    folder = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        if file.endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(folder, file)
            frames = extract_frames(video_path, max_frames=SEQUENCE_LENGTH)
            X.append(frames)
            y.append(label)
            print(f"✅ Processed: {file} ({label})")

X = np.array(X)
y = np.array(y)

print(f"\n📊 Dataset loaded: {X.shape[0]} videos")
print("🔤 Encoding labels ...")
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)
joblib.dump(encoder, ENCODER_PATH)

print("📊 Splitting train / validation sets ...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)

# ======================================================
# 🧠 Build model
# ======================================================
base_model = MobileNetV2(include_top=False, weights="imagenet", pooling=None)
base_model.trainable = False  # Freeze CNN layers

model = Sequential([
    TimeDistributed(base_model, input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),
    TimeDistributed(GlobalAveragePooling2D()),       # ✅ fixes LSTM dimension mismatch
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(len(np.unique(y)), activation="softmax")
])

model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ======================================================
# 🏋️ Train model
# ======================================================
os.makedirs(MODEL_DIR, exist_ok=True)

checkpoint_cb = ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
earlystop_cb = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint_cb, earlystop_cb],
    verbose=1
)

print(f"\n✅ Training complete.")
print(f"✅ Best model saved at: {MODEL_PATH}")
print(f"✅ Label encoder saved at: {ENCODER_PATH}")
