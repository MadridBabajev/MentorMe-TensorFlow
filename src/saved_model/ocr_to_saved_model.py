import tensorflow as tf
import os

os.environ["TF_USE_LEGACY_KERAS"] = "True"

MODEL_DIR = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/input_models/ocr_model_v4.h5"
OUTPUT_DIR = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/output_models/saved_model/ocr"

def main():
    # 1) Load your Keras model from .h5
    model = tf.keras.models.load_model(MODEL_DIR, compile=False)

    # 2) Save it as a SavedModel directory
    tf.saved_model.save(model, OUTPUT_DIR)
    print(f"✅ OCR SavedModel exported to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()