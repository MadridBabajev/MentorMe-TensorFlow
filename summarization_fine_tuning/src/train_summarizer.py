import os
os.environ["TF_USE_LEGACY_KERAS"] = "True"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
print("TF_USE_LEGACY_KERAS:", os.environ.get("TF_USE_LEGACY_KERAS"))
print("TF_GPU_ALLOCATOR:", os.environ.get("TF_GPU_ALLOCATOR"))

import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tensorflow.keras.optimizers.legacy import Adam

# Enable GPU memory growth (optional but often helps)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

# -------------------------------------------
# Custom Callback for Logging (adjust logging frequency)
# -------------------------------------------
class CustomLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Print the positive value of the loss for clarity
        print(f"Epoch {epoch+1} finished. Loss: {-logs.get('loss'):.4f}, Val Loss: {-logs.get('val_loss'):.4f}")

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if batch % 100 == 0:
            print(f"  Batch {batch}: Loss = {logs.get('loss'):.4f}")

# -------------------------------------------
# Custom Callback for Checkpointing (model and optimizer)
# -------------------------------------------
class CustomTFCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager):
        super().__init__()
        self.checkpoint_manager = checkpoint_manager

    def on_epoch_end(self, epoch, logs=None):
        save_path = self.checkpoint_manager.save()
        print(f"Checkpoint saved at: {save_path}")

def train_summarization(batch_size=2, learning_rate=5e-5, epochs=5,
                        max_input_length=512, max_target_length=150,
                        checkpoint_enabled=False, checkpoint_to_load="",
                        initial_epoch=0):
    # -------------------------------------------
    # Model & Tokenizer Setup
    # -------------------------------------------
    model_name = "t5-base"  # or "t5-small" for a smaller model footprint
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    # -------------------------------------------
    # Load the WikiSum Dataset
    # -------------------------------------------
    print("Loading WikiSum dataset...")
    dataset = load_dataset("d0rj/wikisum")

    # -------------------------------------------
    # Preprocessing: Tokenization Function
    # -------------------------------------------
    def preprocess_function(examples):
        inputs = examples["article"]
        targets = examples["summary"]
        model_inputs = tokenizer(inputs, max_length=max_input_length,
                                 truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length,
                               truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Preprocessing training data...")
    train_dataset = dataset["train"].map(preprocess_function)
    print("Preprocessing validation data...")
    val_dataset = dataset["validation"].map(preprocess_function)

    # Set the format for TF conversion.
    columns = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="tensorflow", columns=columns)
    val_dataset.set_format(type="tensorflow", columns=columns)

    def gen_dataset(ds):
        for example in ds:
            yield example, example["labels"]

    output_types = (
        {"input_ids": tf.int32, "attention_mask": tf.int32, "labels": tf.int32},
        tf.int32
    )
    output_shapes = (
        {
            "input_ids": tf.TensorShape([max_input_length]),
            "attention_mask": tf.TensorShape([max_input_length]),
            "labels": tf.TensorShape([max_target_length])
        },
        tf.TensorShape([max_target_length])
    )

    train_tf_dataset = tf.data.Dataset.from_generator(
        lambda: gen_dataset(train_dataset),
        output_types=output_types,
        output_shapes=output_shapes
    ).shuffle(1000).repeat().batch(batch_size)

    val_tf_dataset = tf.data.Dataset.from_generator(
        lambda: gen_dataset(val_dataset),
        output_types=output_types,
        output_shapes=output_shapes
    ).batch(batch_size)

    steps_per_epoch = len(train_dataset) // batch_size
    validation_steps = len(val_dataset) // batch_size

    # -------------------------------------------
    # Compile the Model and Setup Optimizer
    # -------------------------------------------
    optimizer = Adam(learning_rate=learning_rate, decay=1e-4)
    model.compile(optimizer=optimizer, run_eagerly=False)

    # -------------------------------------------
    # Set Up TensorFlow Checkpoint for Model & Optimizer
    # -------------------------------------------
    checkpoint_dir = "./tf_checkpoints"
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)

    # If checkpoint is enabled and exists, restore it.
    if checkpoint_enabled and os.path.exists(checkpoint_to_load + ".index"):
        print(f"Restoring from checkpoint: {checkpoint_to_load}")
        checkpoint.restore(checkpoint_to_load)
        print("Checkpoint restored.")

    # -------------------------------------------
    # Callbacks for Logging, TensorBoard, and Checkpointing
    # -------------------------------------------
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)
    custom_logging = CustomLoggingCallback()
    custom_checkpoint = CustomTFCheckpointCallback(checkpoint_manager)
    callbacks = [tensorboard_callback, custom_logging, custom_checkpoint]

    # -------------------------------------------
    # Training Loop
    # -------------------------------------------
    print("Starting fine-tuning of the Transformer summarization model...")
    model.fit(
        train_tf_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_tf_dataset,
        validation_steps=validation_steps,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )

    # -------------------------------------------
    # Save the Fine-Tuned Model and Tokenizer
    # -------------------------------------------
    save_path = "./fine_tuned_summarizer_v2.keras"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Fine-tuned summarization model saved to {save_path}")

if __name__ == "__main__":
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))

    if tf.config.list_physical_devices('GPU'):
        train_summarization(
            batch_size=2,
            learning_rate=5e-5,
            epochs=1,
            max_input_length=512,
            max_target_length=150,
            checkpoint_enabled=False,
            checkpoint_to_load="./tf_checkpoints/ckpt-1",
            initial_epoch=0
        )