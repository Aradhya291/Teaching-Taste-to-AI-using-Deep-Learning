# model_train.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

DATA = "data/taste_data_5class.csv"
if not os.path.exists(DATA):
    raise FileNotFoundError("❌ Dataset not found. Run prepare_dataset.py first.")

df = pd.read_csv(DATA)
feature_cols = [c for c in df.columns if c not in ["smiles", "taste"]]
X = df[feature_cols].values
y_text = df["taste"]

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(), Dropout(0.35),
    Dense(256, activation='relu'),
    BatchNormalization(), Dropout(0.35),
    Dense(128, activation='relu'),
    BatchNormalization(), Dropout(0.25),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

os.makedirs("artifacts", exist_ok=True)
ckpt = ModelCheckpoint("artifacts/best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
es = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100, batch_size=64,
    callbacks=[ckpt, es], verbose=1
)

# 🔒 Export as TensorFlow SavedModel (bypasses Keras deserialization issues)
export_dir = "artifacts/final_model"  # a folder
# Clean re-export if exists
if os.path.exists(export_dir):
    import shutil; shutil.rmtree(export_dir)
model.export(export_dir)

# Also save metadata needed by the app
np.save("artifacts/label_classes.npy", encoder.classes_)
np.save("artifacts/scaler_mean.npy", scaler.mean_)
np.save("artifacts/scaler_scale.npy", scaler.scale_)
pd.Series(feature_cols).to_csv("artifacts/feature_columns.txt", index=False, header=False)

# Plots
plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.tight_layout(); plt.savefig("artifacts/accuracy.png", dpi=160); plt.close()

plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.tight_layout(); plt.savefig("artifacts/loss.png", dpi=160); plt.close()

print("✅ Exported SavedModel to artifacts/final_model and saved metadata.")

# -------------------------
# 7️⃣ Confusion Matrix
# -------------------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("🧮 Generating confusion matrix...")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred, labels=range(len(np.unique(y))))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("artifacts/confusion_matrix.png", dpi=160)
plt.close()

print("✅ Confusion matrix saved to artifacts/confusion_matrix.png")

