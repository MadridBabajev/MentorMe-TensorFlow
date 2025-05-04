import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

SAVED_MODEL = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/output_models/saved_model/summarization"
TOKENIZER   = "/mnt/c/Users/madri/dev/TensorFlow/ModelConversion/input_models/summarizer"

# ── CONFIG ───────────────────────────────
MAX_ENC     = 512
MAX_DEC     = 150
MAX_GEN     = 120

# ── LOAD ─────────────────────────────────
loaded = tf.saved_model.load(SAVED_MODEL)
infer  = loaded.signatures["serving_default"]

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

# ── ENCODE ONCE ───────────────────────────
text = "summarize: Do not shuck or wash your oysters. Oysters taste best when you shuck them immediately before eating them. In addition, keeping oysters in their shells makes them easier to store and reduces the chance that they'll go bad. If your oysters came pre-shucked in a plastic container, store them in the freezer until you're ready to use them. Leave the grit and dirt on the oysters. This will keep them moist and will help to insulate the meat. Pour ice into a small bowl or other open-top container. Grab a bowl, small cooler, or similar container that you can place inside your fridge. Make sure this container has an open top or removable lid. Then, pour a layer of ice into the bottom of the container. Do not keep your oysters in a sealed or closed-top container. Doing so will suffocate them. You may need to change your ice during the refrigeration process, so do not pour any into the container if you won't be able to check your oysters regularly. Place your oysters on top of the ice bed deep side down. Just like seafood merchants, you'll be storing your oysters on ice to keep them as chilled and fresh as possible. Make sure to turn each of your oysters so that the deeper side faces down, a technique that will help them better retain their juices. Dampen a towel with cold water and place it on top of the oysters. Dip a thin, clean kitchen towel in cold water and ring out the excess liquid. Then, gently lay the towel on top of the oysters. This will keep the oysters from drying out while preventing fresh water poisoning. If you'd prefer, you can cover the oysters with damp paper towels or newspaper instead. Oysters are salt water creatures, so submerging them in fresh water will essentially poison them and lead to their death. Place your container in a refrigerator. If possible, set your refrigerator to a temperature between 35 and 40 °F (2 and 4 °C). Make sure to store your oysters above any raw meat so the juices don't drip down onto your shellfish."
enc = tokenizer(
    text,
    return_tensors="tf",
    padding="max_length", truncation=True,
    max_length=MAX_ENC
)
enc_ids  = enc["input_ids"]
enc_mask = enc["attention_mask"]

# ── PREPARE DECODER BUFFER ─────────────────
pad_id = tokenizer.pad_token_id or 0
eos_id = tokenizer.eos_token_id or 1

# start with all pad tokens (T5 uses pad_token as decoder_start)
dec_buffer = tf.fill([1, MAX_DEC], pad_id)

generated = []

# ── GREEDY ITERATIVE DECODE ────────────────
for step in range(MAX_GEN):
    out = infer(
        encoder_input_ids      = enc_ids,
        encoder_attention_mask = enc_mask,
        decoder_input_ids      = dec_buffer,
    )
    # logits shape = [1, MAX_DEC, vocab_size]
    logits = out["logits"].numpy()
    next_token = int(np.argmax(logits[0, step, :]))
    if next_token == eos_id:
        break
    generated.append(next_token)
    # write it back into decoder buffer
    dec_buffer = tf.tensor_scatter_nd_update(
        dec_buffer,
        [[0, step+1]],
        [next_token]
    )

# ── DECODE & PRINT ─────────────────────────
summary = tokenizer.decode(generated, skip_special_tokens=True)
print("SavedModel summary:", summary)

