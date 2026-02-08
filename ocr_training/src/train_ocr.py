import os
os.environ["TF_USE_LEGACY_KERAS"] = "True"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
print("TF_USE_LEGACY_KERAS:", os.environ.get("TF_USE_LEGACY_KERAS"))
print("TF_GPU_ALLOCATOR:", os.environ.get("TF_GPU_ALLOCATOR"))

import tensorflow as tf
from tensorflow.keras.models import load_model
import time

from ocr_dataset import IAMDataset
from ocr_model import build_crnn_model, ctc_lambda_func
from utils import (extract_vocabulary_from_transcriptions,
                   text_batch_to_indices,
                   ctc_greedy_decode,
                   decode_ids_to_strs,
                   model_log)

# -------------------------------------------
# Vocabulary Extraction from Transcriptions. Local variables
# -------------------------------------------
lines_file = "/mnt/c/Users/madri/dev/TensorFlow/OCR/Project/raw_data/sentences.txt"
base_dir = "/mnt/c/Users/madri/dev/TensorFlow/OCR/Project/raw_data/sentences"
new_model_name = "ocr_model_v4.h5"
log_file = "ocr_training_log.txt"
checkpoint_dir = "./ocr_checkpoints"

# Extract a custom vocabulary from the text corpus
char_list = extract_vocabulary_from_transcriptions(lines_file)
char_to_idx = {ch: i+1 for i, ch in enumerate(char_list)}  # indices starting at 1 (0 reserved)
vocab_size = len(char_list) + 1  # +1 for the CTC blank token
idx_to_char = {v: k for k, v in char_to_idx.items()}

print("Extracted vocabulary:", char_list)
print("Vocabulary size:", vocab_size)

# -------------------------------------------
# The Main Training Function for OCR with Checkpointing
# -------------------------------------------
def train_ocr(
        batch_size=32,
        initial_lr=0.001,
        decay_steps=5000,
        decay_rate=0.95,
        num_epochs=50,
        img_height=128,
        fixed_width=800,
        resume_from_ckpt=True,
        checkpoint_to_load="",
):
    # Open the log file for writing training logs.
    log_f = open(log_file, "w")

    # Step 1: Create the dataset.
    model_log("Loading the dataset...", log_f)
    dataset = IAMDataset(lines_file, base_dir, batch_size=batch_size, img_height=img_height, img_width=fixed_width)

    # Step 2: Build or load the model.
    model_log("Initializing the OCR model...", log_f)
    model = build_crnn_model(img_height=img_height, img_width=fixed_width, vocab_size=vocab_size)

    # Step 3: Set up the optimizer and learning rate schedule.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    # Step 4: Set up TensorFlow Checkpoint for Model, Optimizer, and Global Epoch.
    global_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, global_epoch=global_epoch)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)

    # Restore checkpoint if available using direct file reference.
    start_epoch = 0
    if resume_from_ckpt and os.path.exists(checkpoint_to_load + ".index"):
        model_log(f"Restoring from checkpoint: {checkpoint_to_load}", log_f)
        checkpoint.restore(checkpoint_to_load)
        start_epoch = int(global_epoch.numpy())
        model_log("Checkpoint restored.", log_f)
    else:
        model_log("No checkpoint found. Initializing a new training.", log_f)

    # Step 5: Begin the training loop.
    model_log(f"=== Initializing training ===", log_f)
    steps_per_epoch = len(dataset)
    model_log(f"Steps per epoch: {steps_per_epoch}", log_f)

    for epoch in range(start_epoch, num_epochs):
        model_log(f"Epoch {epoch+1}/{num_epochs}", log_f)
        last_log_time = time.time()
        for step in range(steps_per_epoch):
            x_batch, y_batch_str, paths = dataset[step]
            # Convert label strings to integer sequences.
            y_batch_seq_np = text_batch_to_indices(y_batch_str, char_to_idx, max_length=64)
            y_batch_seq = tf.constant(y_batch_seq_np, dtype=tf.int32)

            # Update model weights.
            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                loss_val = ctc_lambda_func(y_batch_seq, y_pred, vocab_size)
            grads = tape.gradient(loss_val, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 25 == 0:
                current_time = time.time()
                elapsed_time = current_time - last_log_time
                last_log_time = current_time
                y_pred_32 = tf.cast(y_pred, tf.float32)
                decoded_ids = ctc_greedy_decode(y_pred_32, blank_index=vocab_size - 1)
                decoded_texts = decode_ids_to_strs(decoded_ids, idx_to_char)

                total_chars = 0
                match_chars = 0
                for ref_text, hyp_text in zip(y_batch_str, decoded_texts):
                    length = min(len(ref_text), len(hyp_text))
                    for i in range(length):
                        if ref_text[i] == hyp_text[i]:
                            match_chars += 1
                    total_chars += length
                char_acc = (match_chars / total_chars) if total_chars > 0 else 0.0

                log_separator = "=" * 80
                log_msg = (f"Epoch {epoch+1}; Step {step} of {steps_per_epoch}; Image path: {paths[0]}\n"
                           f"Elapsed Time={elapsed_time:.2f}s; CharAcc={char_acc*100:.2f}%;\n"
                           f"Loss={loss_val.numpy():.4f};\n"
                           f"Label:   \"{y_batch_str[0]}\"\n"
                           f"Decoded: \"{decoded_texts[0]}\"\n"
                           f"{log_separator}")
                model_log(log_msg, log_f)
                log_f.flush()

        # End of epoch: update global epoch and save checkpoint.
        global_epoch.assign(epoch + 1)
        ckpt_save_path = checkpoint_manager.save()
        model_log(f"Checkpoint saved at: {ckpt_save_path}", log_f)

    # Step 6: Save the final trained model.
    model.save(new_model_name)
    log_msg = (f"Training complete!\n"
               f"{model.summary()} \n"
               f"Model saved to: {new_model_name}")
    model_log(log_msg, log_f)
    log_f.close()

# -------------------------------------------
# Entry Point
# -------------------------------------------
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    print(tf.__version__)
    print(gpus)

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    if gpus:
        train_ocr(
            batch_size=8,
            initial_lr=0.001,
            decay_steps=5000,
            decay_rate=0.85,
            num_epochs=500,
            img_height=128,
            fixed_width=800,
            resume_from_ckpt=False,
            checkpoint_to_load="./ocr_checkpoints/ckpt-1",
        )