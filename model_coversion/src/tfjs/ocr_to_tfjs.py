import subprocess
import sys

INPUT_MODEL = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/input_models/ocr_model_v4.h5"
OUTPUT_DIR = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/output_models/tfjs/ocr"

def main():
    command = [
        "tensorflowjs_converter",
        "--input_format=keras",
        "--output_format=tfjs_layers_model",
        "--quantize_float16",
        "--",
        INPUT_MODEL,
        OUTPUT_DIR
    ]

    print("Running command:", " ".join(command))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("Error during conversion:")
        print(result.stderr)
        sys.exit(result.returncode)
    else:
        print("✅ Model conversion successful. Output saved to", OUTPUT_DIR)
        print(result.stdout)

if __name__ == "__main__":
    main()
