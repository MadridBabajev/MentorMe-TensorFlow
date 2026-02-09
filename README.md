# MentorMe TensorFlow

This repository contains the **Python/TensorFlow tooling** used to train/fine-tune and convert the two MentorMe AI models:

- **OCR (handwritten text recognition)** — CRNN + CTC loss (TensorFlow/Keras)
- **Summarization** — fine-tuned **T5** (Hugging Face Transformers + TensorFlow)

The scripts here are meant to produce deployment-ready artifacts for different MentorMe applications (backend, web, mobile):
**Keras → SavedModel / TF Lite / TF.js**.

> ⚠️ Most scripts currently use **hard-coded paths** (e.g. `/mnt/c/...` from WSL).
> Before running anything, update the `*_PATH`, `INPUT_MODEL`, `OUTPUT_DIR`, `LINES_FILE`, etc. constants at the top of each script.

---

## Table of Contents

- [Repository contents](#repository-contents)
- [Model formats](#model-formats)
- [Setup](#setup)
- [Training](#training)
    - [Train OCR model](#train-ocr-model)
    - [Fine-tune summarization model](#fine-tune-summarization-model)
- [Testing / benchmarking](#testing--benchmarking)
- [Model conversion](#model-conversion)
    - [OCR conversions](#ocr-conversions)
    - [Summarization conversions](#summarization-conversions)
- [Integration notes](#integration-notes)
- [Troubleshooting](#troubleshooting)
- [Author](#author)

---

## Repository contents

```
.
├─ ocr_training/
│  ├─ train_ocr.py              # train CRNN OCR model + checkpoints
│  ├─ test.py                   # measure OCR inference metrics + CSV output
│  ├─ ocr_dataset.py            # IAM-like dataset loader + preprocessing
│  ├─ ocr_model.py              # CRNN model definition + CTC loss
│  └─ utils.py                  # vocabulary extraction + CTC decoding helpers
│
├─ summarization_training/
│  ├─ train_summarizer.py       # fine-tune T5 (TF) + save_pretrained()
│  └─ test.py                   # measure summarizer inference + ROUGE + CSV output
│
└─ model_conversion/
   ├─ saved_model/
   │  ├─ ocr_to_saved_model.py  # Keras .h5 → SavedModel directory
   │  ├─ t5_to_saved_model.py   # HF TF model → SavedModel with custom signature
   │  ├─ check_model_signature.py
   │  └─ test_t5.py             # quick SavedModel inference sanity check (T5)
   │
   ├─ tflite/
   │  ├─ ocr_to_tflite.py       # Keras .h5 → .tflite (float16 quantization)
   │  ├─ t5_to_tflite.py        # SavedModel → .tflite
   │  └─ test_t5.py             # quick TFLite inference sanity check (T5)
   │
   ├─ tfjs/
   │  ├─ ocr_to_tfjs.py         # Keras .h5 → TF.js layers model
   │  └─ t5_to_tfjs.py          # HF TF model → SavedModel → TF.js graph model
   │
   └─ utils/
      ├─ dataset.py
      ├─ ocr_model_structure.py
      ├─ representative_dataset_ocr.py
      └─ utils.py
```

---

## Model formats

This repo uses **multiple output formats**, depending on where the model must run:

| Format                                         | What it is                    | Typical use                                                     |
|------------------------------------------------|-------------------------------|-----------------------------------------------------------------|
| **Keras** (`.h5` / `.keras`)                   | Simple TF/Keras serialization | training checkpoints, research runs                             |
| **SavedModel** (folder with `.pb` + variables) | Native TF deployment format   | TF Serving, server-side integration, base for other conversions |
| **TF Lite** (`.tflite`)                        | Mobile/edge format            | Flutter / Android / iOS on-device inference                     |
| **TF.js** (`model.json` + `.bin` shards)       | Browser format                | Web inference                                                   |

> **Tokenizer artifacts matter:** the T5 summarization model requires a tokenizer directory (e.g. `config.json`, tokenizer files).

---

## Setup

### Prerequisites

- Python (recommended: **3.10**)
- TensorFlow (many scripts were developed with **TF 2.x** and use legacy Keras compatibility)
- Optional:
    - NVIDIA GPU + CUDA/WSL2 for faster training/conversion
    - On Windows without CUDA, you can try DirectML support

### Create and activate a virtual environment

**Linux / macOS / WSL2**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### Install dependencies

```bash
pip install "tensorflow==2.*" transformers datasets evaluate sentencepiece opencv-python psutil gputil tensorflowjs
```

Notes:
- `tensorflowjs` provides the `tensorflowjs_converter` CLI used by `conversion-src/tfjs/ocr_to_tfjs.py`.
- If you encounter Keras 3 / legacy-Keras issues, see [Troubleshooting](#troubleshooting).

### Quick environment checks

```bash
python --version
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

GPU (optional, but recommended):
```bash
nvidia-smi
nvcc --version
```

---

## Training

### Train OCR model

**Script:** `ocr_training/src/train_ocr.py`

What it does:
- Builds a **CRNN** model (CNN → BiLSTM → logits)
- Uses **CTC loss** for alignment-free sequence learning
- Saves checkpoints to `./ocr_checkpoints`
- Saves the final model to a `.h5` file

#### 1) Configure paths

At the top of `ocr_training/src/train_ocr.py`, update:
- `lines_file` → path to `sentences.txt` (IAM-style metadata)
- `base_dir` → root folder containing the images referenced by the metadata
- `new_model_name` → output `.h5` filename
- `checkpoint_dir` → checkpoints output folder

#### 2) Run training

Run from **inside** `ocr_training/src` so local imports resolve:

```bash
cd ocr-src
python -u train_ocr.py
```

Logging (Linux/WSL2):
```bash
python -u train_ocr.py | tee ocr_training_log.txt
```

---

### Fine-tune summarization model

**Script:** `summarization_fine_tuning/src/train_summarizer.py`

What it does:
- Loads **T5** (`t5-base` by default)
- Fine-tunes on the **WikiSum** dataset via `datasets`
- Writes TensorBoard logs to `./logs`
- Saves the fine-tuned model + tokenizer using `save_pretrained()` to:
    - `./fine_tuned_summarizer_v2.keras` (folder)

Run from inside `summarization_fine_tuning/src`:

```bash
cd sum-src
python -u train_summarizer.py
```

TensorBoard:
```bash
tensorboard --logdir ./logs
```

---

## Testing / benchmarking

### OCR benchmark

**Script:** `ocr_training/src/test.py`

Update config constants at the top:
- `MODEL_PATH`, `LINES_FILE`, `BASE_DIR`, etc.

Run:
```bash
cd ocr-src
python -u test.py
```

Outputs:
- prints load time, inference time, CPU/RAM usage
- writes `inference_metrics.csv`

### Summarizer benchmark

**Script:** `summarization_fine_tuning/src/test.py`

Update config constants at the top:
- `MODEL_PATH` (tokenizer + model directory)
- `TEXT` and `REFERENCE_SUMMARY`

Run:
```bash
cd sum-src
python -u test.py
```

Outputs:
- prints load time, inference time, CPU/RAM/GPU usage
- prints ROUGE metrics
- writes `summarizer_metrics.csv`

---

## Model conversion

> ✅ Tip: keep a consistent local folder layout, for example:
>
> ```
> models/
>   input/
>     ocr_model.h5
>     summarizer/
>   output/
>     saved_model/
>     tflite/
>     tfjs/
> ```
>
> Then update the `INPUT_MODEL`, `MODEL_DIR`, `OUTPUT_DIR`, etc. constants in each script.

### OCR conversions

#### Keras `.h5` → SavedModel

```bash
python model_coversion/src/saved_model/ocr_to_saved_model.py
```

#### Keras `.h5` → TF Lite (`.tflite`)

This script uses:
- `Optimize.DEFAULT`
- `float16` post-training quantization
- a representative dataset generator for calibration

```bash
python conversion-src/tflite/ocr_to_tflite.py
```

#### Keras `.h5` → TF.js (layers model)

Uses the `tensorflowjs_converter` CLI:

```bash
python model_coversion/src/tfjs/ocr_to_tfjs.py
```

Outputs:
- `model.json`
- one or more weight shard files (`.bin`)

---

### Summarization conversions

#### HF TF model → SavedModel (custom signature)

T5 is wrapped in a `tf.Module` to export a stable serving signature that includes decoder inputs.

```bash
python conversion-src/saved_model/t5_to_saved_model.py
```

You can inspect the produced signature:

```bash
python conversion-src/saved_model/check_model_signature.py
```

#### SavedModel → TF Lite

```bash
python conversion-src/tflite/t5_to_tflite.py
```

#### HF TF model → TF.js (graph model)

This script:
1) wraps T5 in a `tf.Module`
2) exports a SavedModel
3) converts it to TF.js artifacts

```bash
python conversion-src/tfjs/t5_to_tfjs.py
```

---

## Integration notes

### Backend / server-side (SavedModel)

SavedModel is the most stable “server-friendly” artifact:
- Use it for TensorFlow Serving, Python inference services, or server-side integration libraries.
- Keep tokenizers (T5) and post-processing (CTC decode) close to the model.

### React (TF.js)

- OCR export (`tfjs_layers_model`) is typically loaded with `tf.loadLayersModel()`.
- T5 TF.js export is produced via SavedModel conversion and is typically loaded as a graph model (`tf.loadGraphModel()`).

> Real-world T5 summarization fully in-browser is often heavy (model size + iterative decoding).
> Consider server-side summarization or smaller distilled models if you need web inference.

### Flutter (TF Lite)

- OCR `.tflite` is suitable for on-device inference, but you still need:
    - the same image preprocessing (resize/pad/normalize)
    - a CTC decoding implementation
- T5 `.tflite` may be too large/slow for many mobile devices; measure before committing.

---

## Troubleshooting

### Keras 3 vs legacy Keras

Several scripts set:
```python
os.environ["TF_USE_LEGACY_KERAS"] = "True"
```

If you see errors related to `keras` vs `tf.keras`, keep this enabled, and prefer TensorFlow 2.x.

### TF Lite converter issues

If TF Lite conversion fails in your current environment:
- try **Python 3.10** and a stable TF 2.x version
- keep the conversion environment separate from your training environment if needed

### Windows / WSL utility commands (from the original dev notes)

Check TF and GPU:
```bash
python --version
pip freeze | grep -E "tensorflow"
nvcc --version
nvidia-smi
```

Reinstall TF (example pattern):
```bash
pip cache purge
pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel tensorflow-estimator tensorflow-io-gcs-filesystem
pip install --upgrade --force-reinstall "tensorflow==2.12.*"
```

---

_Madrid Babajev (08.02.2026)_
