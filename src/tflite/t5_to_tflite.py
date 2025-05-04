import tensorflow as tf

SAVED_MODEL   = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/output_models/saved_model/summarization"
TFLITE_OUT    = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/output_models/tflite/summarization/summarization_model.tflite"

def main():
    conv = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL, signature_keys=["serving_default"])
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = []
    conv.experimental_enable_resource_variables = True

    tflite_model = conv.convert()
    open(TFLITE_OUT, "wb").write(tflite_model)
    print("✅ TFLite model written to:", TFLITE_OUT)

if __name__ == "__main__":
    main()