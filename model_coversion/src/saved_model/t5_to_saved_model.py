import pathlib

import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM
import os

os.environ["TF_USE_LEGACY_KERAS"] = "True"

MODEL_DIR = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/input_models/summarizer"
OUT_DIR = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/output_models/saved_model/summarization"

MAX_ENC = 512
MAX_DEC = 150


class T5Export(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    @tf.function(
        input_signature=[
            tf.TensorSpec([None, MAX_ENC], tf.int32, name="encoder_input_ids"),
            tf.TensorSpec([None, MAX_ENC], tf.int32, name="encoder_attention_mask"),
            tf.TensorSpec([None, MAX_DEC], tf.int32, name="decoder_input_ids"),
        ]
    )
    def __call__(self,
                 encoder_input_ids: tf.Tensor,
                 encoder_attention_mask: tf.Tensor,
                 decoder_input_ids: tf.Tensor):
        out = self.model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            training=False,
        )
        return {"logits": out.logits}


def main():
    # Load TensorFlow checkpoint
    model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    exporter = T5Export(model)

    # Get a ConcreteFunction ONCE from that same object
    concrete_fn = exporter.__call__.get_concrete_function()

    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(
        exporter,
        OUT_DIR,
        signatures={"serving_default": concrete_fn}
    )
    print("✅ SavedModel exported to:", OUT_DIR)


if __name__ == "__main__":
    main()
