import sys
import os

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.insert(0, project_root)

import tensorflow as tf
from src.utils.representative_dataset_ocr import representative_data_gen

# os.environ["TF_USE_LEGACY_KERAS"] = "True"

INPUT_MODEL = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/input_models/ocr_model_v4.h5"
OUTPUT_MODEL = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/output_models/tflite/ocr/ocr_model.tflite"
LINES_FILE = "/mnt/c/Users/madri/dev/TensorFlow/OCR/Project/raw_data/sentences.txt"
BASE_DIR   = "/mnt/c/Users/madri/dev/TensorFlow/OCR/Project/raw_data/sentences"

def main():
    model = tf.keras.models.load_model(INPUT_MODEL, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Provide dataset for calibration
    converter.representative_dataset = lambda: representative_data_gen(
        lines_file=LINES_FILE,
        base_dir=BASE_DIR,
        sample_count=100
    )
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    with open(OUTPUT_MODEL, "wb") as f:
        f.write(tflite_model)
    print("✅ TensorFlow Lite model exported successfully as ocr_model.tflite")

if __name__ == "__main__":
    main()