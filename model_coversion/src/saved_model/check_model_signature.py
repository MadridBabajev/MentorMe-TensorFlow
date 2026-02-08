import tensorflow as tf

MODEL_PATH = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/output_models/saved_model/summarization" #"/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/output_models/saved_model/ocr"

model = tf.saved_model.load(MODEL_PATH)
fn = model.signatures["serving_default"]
print("Inputs :", fn.structured_input_signature)
print("Outputs:", fn.structured_outputs)
