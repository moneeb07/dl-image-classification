"""
STL-10 Image Classifier — Streamlit Demo
CS-419 Deep Learning Project
Best model: reg_He_BN (He uniform init + BatchNormalization), test acc 69.3%
"""

import os
import io
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STL-10 Classifier — CS419 DL Project",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")

CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

IMG_SIZE = 96

# Map a friendly label → filename in models/
AVAILABLE_MODELS = {
    "Best: He + BatchNorm (69.3%)":      "reg_He_BN_best.keras",
    "Deep CNN":                          "deep_CNN_best.keras",
    "He + Dropout + Aug":                "reg_He_BN_Drop_Aug_best.keras",
    "He baseline":                       "reg_He_baseline_best.keras",
    "Glorot baseline":                   "reg_Glorot_baseline_best.keras",
    "CNN baseline":                      "CNN_baseline_best.keras",
    "MLP baseline":                      "MLP_baseline_best.keras",
    "Activation: ELU":                   "act_ELU_best.keras",
    "Activation: ReLU":                  "act_ReLU_best.keras",
    "Activation: LeakyReLU":             "act_LeakyReLU_best.keras",
    "Activation: Swish":                 "act_Swish_best.keras",
    "Optimizer: Adam":                   "opt_Adam_best.keras",
    "Optimizer: RMSprop":                "opt_RMSprop_best.keras",
    "Optimizer: SGD":                    "opt_SGD_best.keras",
    "Optimizer: SGD + Momentum":         "opt_SGD_momentum_best.keras",
}

DEFAULT_MODEL_LABEL = "Best: He + BatchNorm (69.3%)"


# ─────────────────────────────────────────────────────────────────────
# Model loader (cached)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(model_filename: str):
    path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(path):
        return None
    return keras.models.load_model(path, compile=False)


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Resize → normalize → add batch dim. Matches notebook preprocessing."""
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────────────
# Sidebar — model + navigation
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("Settings")

model_label = st.sidebar.selectbox(
    "Model",
    list(AVAILABLE_MODELS.keys()),
    index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_LABEL),
)
model_file = AVAILABLE_MODELS[model_label]
model_path = os.path.join(MODELS_DIR, model_file)

st.sidebar.markdown("---")
st.sidebar.markdown("**Models directory**")
st.sidebar.code(MODELS_DIR, language=None)

if not os.path.exists(model_path):
    st.sidebar.error(f"Missing: {model_file}")
else:
    st.sidebar.success(f"Loaded: {model_file}")

page = st.sidebar.radio(
    "View",
    ["Predict", "Project Charts", "About"],
    index=0,
)


# ─────────────────────────────────────────────────────────────────────
# PAGE 1: Predict
# ─────────────────────────────────────────────────────────────────────
def page_predict():
    st.title("🧠 STL-10 Image Classifier")
    st.caption(
        "Upload a photo of an airplane, bird, car, cat, deer, dog, horse, "
        "monkey, ship, or truck. The model resizes it to 96×96 and predicts a class."
    )

    if not os.path.exists(model_path):
        st.error(
            f"Model file not found at:\n\n`{model_path}`\n\n"
            f"Place **{model_file}** into the `models/` folder shown in the sidebar."
        )
        return

    model = load_model(model_file)
    if model is None:
        st.error("Failed to load model.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload image")
        uploaded = st.file_uploader(
            "Choose a JPG / PNG image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        if uploaded is not None:
            pil_img = Image.open(io.BytesIO(uploaded.read()))
            st.image(pil_img, caption="Input image", use_container_width=True)
        else:
            pil_img = None
            st.info("Waiting for an image…")

    with col2:
        st.subheader("Prediction")
        if pil_img is None:
            st.write("Upload an image on the left to see predictions.")
            return

        x = preprocess_image(pil_img)
        probs = model.predict(x, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        top_class = CLASS_NAMES[top_idx]
        top_conf = float(probs[top_idx])

        st.metric(label="Top prediction", value=top_class.upper(),
                  delta=f"{top_conf*100:.2f}% confidence")

        st.markdown("**All class probabilities**")
        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        # Sort by probability descending
        sorted_items = sorted(prob_dict.items(), key=lambda kv: -kv[1])
        for cls, p in sorted_items:
            st.progress(min(max(p, 0.0), 1.0), text=f"{cls:<10s}  {p*100:5.2f}%")


# ─────────────────────────────────────────────────────────────────────
# PAGE 2: Project Charts
# ─────────────────────────────────────────────────────────────────────
def page_charts():
    st.title("📊 Project Charts")
    st.caption("Figures produced by the notebook (place .png files in `plots/`).")

    chart_groups = [
        ("Dataset overview", [
            ("sample_images.png",        "Sample images — one per class"),
            ("class_distribution.png",   "Training-set class distribution"),
        ]),
        ("Experiment comparisons", [
            ("cmp_optimizers.png",       "Optimizer comparison"),
            ("cmp_activations.png",      "Activation function comparison"),
            ("cmp_regularization.png",   "Regularization comparison"),
        ]),
        ("Training curves", [
            ("curves_MLP_baseline.png",  "MLP baseline curves"),
            ("curves_CNN_baseline.png",  "CNN baseline curves"),
            ("curves_deep_CNN.png",      "Deep CNN curves"),
            ("curves_transfer_ResNet50.png", "Transfer learning curves"),
        ]),
        ("Final analysis", [
            ("comparative_all_models.png", "All models — test accuracy"),
            ("ablation_study.png",         "Ablation study"),
            ("confusion_matrix.png",       "Confusion matrix (best model)"),
            ("overfitting_analysis.png",   "Overfitting analysis"),
            ("visual_predictions.png",     "Visual prediction samples"),
        ]),
    ]

    any_found = False
    for group_title, items in chart_groups:
        st.markdown(f"### {group_title}")
        cols = st.columns(2)
        idx = 0
        for fname, caption in items:
            fpath = os.path.join(PLOTS_DIR, fname)
            if os.path.exists(fpath):
                any_found = True
                with cols[idx % 2]:
                    st.image(fpath, caption=caption, use_container_width=True)
                idx += 1
            else:
                with cols[idx % 2]:
                    st.warning(f"Missing: `plots/{fname}`")
                idx += 1
        st.markdown("---")

    if not any_found:
        st.info(
            f"No chart images found in `{PLOTS_DIR}`.\n\n"
            "Download the PNGs from your Drive `plots/` folder and place them here."
        )


# ─────────────────────────────────────────────────────────────────────
# PAGE 3: About
# ─────────────────────────────────────────────────────────────────────
def page_about():
    st.title("About this project")
    st.markdown(
        """
**CS-419 Deep Learning — Course Project**
Experimental Study of Deep Learning Techniques for Image Classification on STL-10.

**Group Members**
- Moneeb Ur Rahman — 454015
- Usman Javaid — 470729
- Muhammad Hammad Irfan — 461864

**Dataset:** STL-10 — 10 classes, 96×96 RGB images
(airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck).

**Best model:** `reg_He_BN` — three-block CNN with He uniform initialization
and Batch Normalization. Test accuracy **69.3%** on the held-out 7,000-image
test split.

**Pipeline summary**
1. Resize input to 96×96
2. Normalize pixel values to [0, 1]
3. Forward pass through the saved Keras model
4. Softmax over 10 classes → top-1 prediction

**Files expected**
- `models/reg_He_BN_best.keras` — the best trained model (place from Drive)
- `plots/*.png` — optional figures from the notebook
        """
    )


# ─────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────
if page == "Predict":
    page_predict()
elif page == "Project Charts":
    page_charts()
else:
    page_about()
