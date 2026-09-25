# STL-10 Image Classification: Streamlit Demo

A Streamlit web app that classifies 96×96 images into one of 10 STL-10 categories
using a CNN trained from scratch (He uniform initialization + Batch Normalization).

> **Course:** CS-419 Deep Learning · **Semester:** Spring 2026
> **Institution:** NUST · **Group:** Moneeb Ur Rahman, Usman Javaid, Muhammad Hammad Irfan

---

## Live demo

https://dl-image-classification.streamlit.app/

## Paper & docs

The full report and supporting documentation for this project live in this Drive folder:
https://drive.google.com/drive/folders/1RxZE2aUj7tWyYoDrxX_z1rbaoJj9xTJA?usp=sharing

## Classes

`airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck`

## Best model

| Item | Value |
|---|---|
| Architecture | 3-block CNN, GlobalAveragePooling, Dense head |
| Activation | ELU |
| Initialization | He uniform |
| Regularization | BatchNormalization (no dropout, no L2) |
| Optimizer | Adam (lr=1e-3) |
| Trainable params | 111,498 |
| **Test accuracy** | **69.3%** (on 7,000 held-out STL-10 test images) |

Built after an experimental study across optimizers (SGD vs SGD+momentum vs RMSprop vs Adam),
activations (ReLU / LeakyReLU / ELU / Swish), initializers (Glorot vs He),
regularization (Dropout / L2 / Augmentation), Batch Normalization, a deeper CNN,
and ResNet50 transfer learning. See `plots/comparative_all_models.png` for the
full comparison.

---

## Features

- **Predict tab**: upload an image, get the top-1 class plus full probability bars over all 10 classes.
- **Project Charts tab**: view every figure from the experimental study (optimizer comparison,
  activation comparison, ablation study, confusion matrix, overfitting analysis, etc.).
- **Model switcher**: sidebar dropdown to try any of the 15 trained models if their
  `.keras` files are placed in `models/`.
- **About tab**: project context and pipeline summary.

---

## Run locally

```bash
git clone https://github.com/moneeb07/dl-image-classification.git
cd dl-image-classification

python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).

### Optional: get real STL-10 test images

`get_test_images.py` downloads STL-10 via TensorFlow Datasets and saves a few
test images per class into `test_images/<class_name>/`, useful for verifying the
model against its true distribution.

```bash
pip install tensorflow-datasets
python get_test_images.py
```

> `tensorflow-datasets` is **not** in `requirements.txt` because the deployed
> app doesn't need it, it's only used by this helper.

---

## Deployment

The live demo is hosted on Streamlit Community Cloud, built directly from the `main`
branch (`app.py`). The best model file, `models/reg_He_BN_best.keras` (1.2 MB), is
committed to the repo, so the build needs no extra setup beyond `requirements.txt`.
It runs on Streamlit Cloud's default Python 3.11, and the `tensorflow-cpu` runtime
plus this model fit comfortably within the 1 GB memory limit of the free tier.

---

## Repository layout

```
dl-image-classification/
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies (tensorflow-cpu for Cloud)
├── get_test_images.py          # Helper: download STL-10 test images
├── README.md
├── .gitignore
├── models/
│   └── reg_He_BN_best.keras    # Best model (committed, 1.2 MB)
└── plots/
    ├── sample_images.png
    ├── class_distribution.png
    ├── cmp_optimizers.png
    ├── cmp_activations.png
    ├── cmp_regularization.png
    ├── comparative_all_models.png
    ├── confusion_matrix.png
    ├── ablation_study.png
    ├── overfitting_analysis.png
    ├── visual_predictions.png
    └── curves_*.png            # Training curves per model
```

---

## Inference pipeline

```text
PIL Image → convert RGB → resize to 96×96 → /255.0 → batch dim → model → softmax(10)
```

Matches the preprocessing in the training notebook exactly, so predictions on
STL-10 test images reproduce the reported 69.3% test accuracy.

---

## Limitations

- **Training set was small** (5,000 images, 500 per class), so the model plateaus at
  ~69% from scratch. Errors concentrate on visually similar classes
  (cat/dog/monkey, car/truck/ship).
- **Distribution shift**: STL-10 photos are tightly cropped on the subject.
  Wide-angle phone shots or cluttered backgrounds will get lower confidence.
- **No GPU at inference**: `tensorflow-cpu` is used. A single prediction takes
  ~100 ms on a modest cloud instance.

See `plots/confusion_matrix.png` and `plots/overfitting_analysis.png` for the
full error analysis.

---

## License

Academic project, released as-is for educational purposes.
