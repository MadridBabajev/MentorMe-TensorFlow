import os
import tensorflow as tf
import tensorflowjs as tfjs
from transformers import TFAutoModelForSeq2SeqLM

MODEL_DIR = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/input_models/summarizer"
OUTPUT_DIR = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/output_models/tfjs/summarization"

class T5Export(tf.Module):
    """
    Exports a T5 model that expects:
      - encoder_input_ids (int32)
      - encoder_attention_mask (int32)
      - decoder_input_ids (int32)
    and returns final logits from the decoder.
    """
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    @tf.function(
        input_signature=[
            {
                "encoder_input_ids": tf.TensorSpec([None, None], tf.int32, name="encoder_input_ids"),
                "encoder_attention_mask": tf.TensorSpec([None, None], tf.int32, name="encoder_attention_mask"),
                "decoder_input_ids": tf.TensorSpec([None, None], tf.int32, name="decoder_input_ids"),
            }
        ]
    )
    def forward(self, inputs):
        """
        Single forward pass of T5.
        For a real summarization decode, you'd call T5.generate() in Python,
        or do iterative feeding in JavaScript.
        """
        output = self.hf_model(
            input_ids=inputs["encoder_input_ids"],
            attention_mask=inputs["encoder_attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            training=False
        )
        return {"logits": output.logits}

def main():
    # Ensure the model directory actually exists
    if not os.path.isdir(MODEL_DIR):
        raise ValueError(f"MODEL_DIR should be a folder with config.json, tf_model.h5, etc. Got: {MODEL_DIR}")

    print(f"Loading T5 model from {MODEL_DIR}")
    model = TFAutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR,
        from_pt=False,
        use_cache=False,
    )

    # Wrap in our tf.Module to define a serving signature that includes decoder_input_ids
    export_module = T5Export(model)

    # Export as a SavedModel
    saved_model_path = os.path.join(OUTPUT_DIR, "saved_model")
    print(f"Saving SavedModel to {saved_model_path}")
    tf.saved_model.save(
        export_module,
        saved_model_path,
        signatures={"serving_default": export_module.forward}
    )

    # Convert that SavedModel to TF.js
    tfjs_target_dir = os.path.join(OUTPUT_DIR, "web_model")
    print(f"Converting to TF.js in {tfjs_target_dir} ...")
    tfjs.converters.convert_tf_saved_model(
        saved_model_dir=saved_model_path,
        output_dir=tfjs_target_dir,
        control_flow_v2=True,
        strip_debug_ops=True
    )
    print("✅ Done. TF.js model artifacts are in:", tfjs_target_dir)

if __name__ == "__main__":
    main()