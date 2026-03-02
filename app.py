# app.py
import os, warnings
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors

# Optional: silence TF logs + oneDNN note
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import tensorflow as tf

st.set_page_config(page_title="Taste Classifier", page_icon="🍭")
st.title("🍭 Taste Sensing AI")
st.write("Predicts **Sweet / Bitter / Umami / Sour / Salty** from a SMILES (Simplified Molecular Input Line Entry System) string.")

# ---- Load SavedModel (no Keras deserialization) ----
MODEL_DIR = "artifacts/final_model"
if not os.path.isdir(MODEL_DIR):
    st.error("❌ Model folder not found. Please run:  python model_train.py")
    st.stop()

try:
    loaded = tf.saved_model.load(MODEL_DIR)
    serving_fn = loaded.signatures["serving_default"]
    input_name = list(serving_fn.structured_input_signature[1].keys())[0]
    output_name = list(serving_fn.structured_outputs.keys())[0]
    st.success("✅ Model loaded (SavedModel).")
except Exception as e:
    st.error(f"⚠️ Failed to load SavedModel: {e}")
    st.stop()

# ---- Load preprocessing metadata ----
try:
    label_classes = np.load("artifacts/label_classes.npy", allow_pickle=True)
    scaler_mean = np.load("artifacts/scaler_mean.npy")
    scaler_scale = np.load("artifacts/scaler_scale.npy")
    feature_cols = open("artifacts/feature_columns.txt").read().strip().splitlines()
except Exception:
    st.error("❌ Missing preprocessing files in 'artifacts/'. Re-run training.")
    st.stop()

# ---- Descriptor generator (RDKit) ----
desc_funcs = {d[0]: d[1] for d in Descriptors.descList}

def smiles_to_desc(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    vals = []
    for n in feature_cols:
        f = desc_funcs.get(n)
        try:
            vals.append(f(mol) if f else np.nan)
        except Exception:
            vals.append(np.nan)
    arr = np.array(vals, dtype=float)
    if np.isnan(arr).any():
        return None
    return arr

# ---- UI: single prediction ----
st.subheader("🔬 Predict from SMILES")
smiles = st.text_input("Enter SMILES (e.g., C(C(=O)O)CC(N)C(=O)O for Glutamic Acid / Umami)")

if st.button("Predict Taste"):
    if not smiles.strip():
        st.warning("⚠️ Please enter a SMILES string.")
    else:
        x = smiles_to_desc(smiles.strip())
        if x is None:
            st.error("❌ Invalid SMILES or descriptors could not be computed.")
        else:
            x = (x - scaler_mean) / scaler_scale
            x = x.reshape(1, -1).astype(np.float32)
            # Call SavedModel
            out = serving_fn(**{input_name: tf.constant(x)})
            probs = out[output_name].numpy()[0]
            pred = label_classes[int(np.argmax(probs))]
            st.success(f"### 🧠 Predicted Taste: **{pred.capitalize()}**")
            st.bar_chart(pd.DataFrame(probs, index=label_classes, columns=["Probability"]))

# ---- Training curves ----
st.subheader("📊 Training Performance")
c1, c2 = st.columns(2)
with c1:
    if os.path.exists("artifacts/accuracy.png"):
        st.image("artifacts/accuracy.png", caption="Accuracy")
with c2:
    if os.path.exists("artifacts/loss.png"):
        st.image("artifacts/loss.png", caption="Loss")

# ---- Confusion Matrix ----
st.subheader("🧩 Confusion Matrix (Model Evaluation)")
if os.path.exists("artifacts/confusion_matrix.png"):
    st.image("artifacts/confusion_matrix.png", caption="Confusion Matrix")
else:
    st.info("ℹ️ Confusion matrix not found. Please retrain the model to generate it.")

st.caption("SavedModel loading avoids Keras 3 deserialization bugs (BatchNorm/Sequential).")
