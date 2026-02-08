#!/usr/bin/env python3
import os
import time
import csv
import psutil
import tensorflow as tf
from tensorflow.keras.models import load_model

from ocr_dataset import IAMDataset
from ocr_model import ctc_lambda_func, build_crnn_model
from utils import (
    extract_vocabulary_from_transcriptions,
    text_batch_to_indices,
    ctc_greedy_decode,
    decode_ids_to_strs
)

# ------------------ CONFIGURATION ------------------
MODEL_PATH  = "./ocr_model.h5"
LINES_FILE  = "./sentences.txt"
BASE_DIR    = "./sentences"
BATCH_SIZE  = 1
IMG_HEIGHT  = 128
IMG_WIDTH   = 800
MAX_LENGTH  = 64
CSV_LOG     = "inference_metrics.csv"
# ---------------------------------------------------

def load_keras_model(path, img_height, img_width, vocab_size):
    # 1) Rebuild the exact architecture
    model = build_crnn_model(
        img_height=img_height,
        img_width=img_width,
        vocab_size=vocab_size
    )
    # 2) Load the weights from the .h5
    model.load_weights(path)
    return model

def main():
    # Build vocabulary mappings
    chars = extract_vocabulary_from_transcriptions(LINES_FILE)
    char_to_idx = {ch: i+1 for i, ch in enumerate(chars)}
    vocab_size = len(chars) + 1
    idx_to_char = {i+1: ch for i, ch in enumerate(chars)}

    # Prepare one batch of data
    dataset = IAMDataset(
        lines_file=LINES_FILE,
        base_dir=BASE_DIR,
        batch_size=BATCH_SIZE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        max_length=MAX_LENGTH
    )
    x_batch, y_batch_strs, paths = dataset[0]

    # Process handle for resource measurements
    proc = psutil.Process(os.getpid())

    # --- Load model ---
    print("Loading model...")
    t0 = time.time()
    model = load_keras_model(MODEL_PATH, IMG_HEIGHT, IMG_WIDTH, vocab_size)
    load_time = time.time() - t0
    print(f"Model load time: {load_time:.2f}s\n")

    # Warm-up (TF graph + GPU)
    _ = model.predict(x_batch)

    # --- Measure resources & run inference ---
    mem_before = proc.memory_info().rss / (1024**2)  # in MB
    # initialize CPU percent counter
    psutil.cpu_percent(interval=None)

    t1 = time.time()
    y_pred = model(x_batch, training=False)
    inf_time = time.time() - t1

    cpu_pct = psutil.cpu_percent(interval=None)      # over the inference interval
    mem_after = proc.memory_info().rss / (1024**2)   # in MB
    mem_delta = mem_after - mem_before

    # --- Decode & compute accuracy ---
    logits = tf.cast(y_pred, tf.float32)
    decoded_ids = ctc_greedy_decode(logits, blank_index=vocab_size-1)
    decoded_strs = decode_ids_to_strs(decoded_ids, idx_to_char)

    # Character-level accuracy
    total_chars = 0
    match_chars = 0
    for ref, hyp in zip(y_batch_strs, decoded_strs):
        L = min(len(ref), len(hyp))
        for i in range(L):
            if ref[i] == hyp[i]:
                match_chars += 1
        total_chars += L
    char_acc = (match_chars / total_chars) if total_chars > 0 else 0.0

    # --- Print results ---
    print(f"Inference time      : {inf_time:.4f} s")
    print(f"Memory (before/after): {mem_before:.1f} → {mem_after:.1f} MB  (Δ {mem_delta:.1f} MB)")
    print(f"CPU usage           : {cpu_pct:.1f} %")
    print(f"Char-level accuracy : {char_acc*100:.2f} %\n")

    # --- Log to CSV ---
    with open(CSV_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "load_time_s",
            "inference_time_s",
            "mem_before_MB",
            "mem_after_MB",
            "mem_delta_MB",
            "cpu_percent",
            "char_accuracy"
        ])
        writer.writerow([
            f"{load_time:.4f}",
            f"{inf_time:.4f}",
            f"{mem_before:.2f}",
            f"{mem_after:.2f}",
            f"{mem_delta:.2f}",
            f"{cpu_pct:.2f}",
            f"{char_acc:.4f}"
        ])
    print(f"Metrics written to {CSV_LOG}")

if __name__ == "__main__":
    main()
