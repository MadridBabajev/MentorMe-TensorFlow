#!/usr/bin/env python3
import os
import time
import csv
import psutil
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from evaluate import load
import GPUtil
import tensorflow as tf

# ------------------ CONFIGURATION ------------------
MODEL_PATH        = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/input_models/summarizer"
CSV_LOG           = "summarizer_metrics.csv"
TEXT              = """Text to summarize"""
REFERENCE_SUMMARY = """target summary"""
MAX_INPUT_LEN     = 512
MAX_SUMMARY_LEN   = 150
# ---------------------------------------------------

def main():
    # Prepare resource monitor
    proc = psutil.Process(os.getpid())
    # Load ROUGE metric
    rouge = load("rouge")

    # Load model & tokenizer
    print("Loading model and tokenizer…")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    load_time = time.time() - t0
    print(f"  → Load time: {load_time:.2f}s\n")

    # Pre-encode inputs
    inputs = tokenizer(
        "summarize: " + TEXT,
        return_tensors="tf",
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length"
    )

    # Warm-up
    _ = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=MAX_SUMMARY_LEN,
        num_beams=2,
        early_stopping=True
    )

    # Measure before resources
    mem_before = proc.memory_info().rss / 1024**2
    psutil.cpu_percent(interval=None)  # init counter
    g = GPUtil.getGPUs()[0]
    gpu_mem_before = g.memoryUsed      # MB
    gpu_load_before = g.load * 100     # %

    # Run inference & time it
    t1 = time.time()
    gen_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=MAX_SUMMARY_LEN,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    inf_time = time.time() - t1

    # Measure after resources
    cpu_pct   = psutil.cpu_percent(interval=None)
    mem_after = proc.memory_info().rss / 1024**2
    mem_delta = mem_after - mem_before

    g = GPUtil.getGPUs()[0]
    gpu_mem_after = g.memoryUsed       # MB
    gpu_load_after = g.load * 100      # %

    # Decode
    summary = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # Compute ROUGE
    results = rouge.compute(
        predictions=[summary],
        references=[REFERENCE_SUMMARY],
        rouge_types=["rouge1", "rouge2", "rougeL"]
    )
    # Extract F-scores
    r1 = results["rouge1"]
    r2 = results["rouge2"]
    rL = results["rougeL"]

    # Print metrics
    print("=== Inference Metrics ===")
    print(f"Inference time      : {inf_time:.4f} s")
    print(f"Memory (MB)         : {mem_before:.1f} → {mem_after:.1f}  (Δ {mem_delta:.1f})")
    print(f"CPU usage           : {cpu_pct:.1f} %")
    print(f"GPU memory (MB)     : {gpu_mem_before:.1f} → {gpu_mem_after:.1f}  (Δ {gpu_mem_after-gpu_mem_before:.1f})")
    print(f"GPU utilization     : {gpu_load_after:.1f} %")
    print("\n=== Summarization Quality (ROUGE F1) ===")
    print(f"ROUGE-1 F1          : {r1*100:.2f}%")
    print(f"ROUGE-2 F1          : {r2*100:.2f}%")
    print(f"ROUGE-L F1          : {rL*100:.2f}%\n")

    # Write CSV
    with open(CSV_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "load_time_s", "inference_time_s",
            "mem_before_MB", "mem_after_MB", "mem_delta_MB",
            "gpu_mem_before_MB", "gpu_mem_after_MB", "gpu_mem_delta_MB", "gpu_util_after_pct",
            "cpu_percent",
            "rouge1_f1", "rouge2_f1", "rougeL_f1"
        ])
        writer.writerow([
            f"{load_time:.4f}", f"{inf_time:.4f}",
            f"{mem_before:.2f}", f"{mem_after:.2f}", f"{mem_delta:.2f}",
            f"{gpu_mem_before:.2f}", f"{gpu_mem_after:.2f}",
            f"{gpu_mem_after-gpu_mem_before:.2f}", f"{gpu_load_after:.2f}"
            f"{cpu_pct:.2f}",
            f"{r1:.4f}", f"{r2:.4f}", f"{rL:.4f}"
        ])
    print(f"Metrics logged to {CSV_LOG}")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    main()
