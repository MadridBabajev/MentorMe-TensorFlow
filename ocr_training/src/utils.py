import numpy as np
import tensorflow as tf

def extract_vocabulary_from_transcriptions(vocabulary_file):
    vocab_set = set()
    with open(vocabulary_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            # Transcription is everything after the first 9 fields.
            transcription = " ".join(parts[9:]).replace('|', ' ')
            for ch in transcription:
                vocab_set.add(ch)
    return sorted(vocab_set)

def text_batch_to_indices(label_strs, char_map, max_length=64):
    """
    Convert a list of label strings into a fixed-size numpy array of indices.
    """
    out = np.zeros((len(label_strs), max_length), dtype=np.int32)
    for i, text in enumerate(label_strs):
        for j, ch in enumerate(text):
            if j >= max_length:
                break
            out[i, j] = char_map.get(ch, 0)
    return out

def ctc_greedy_decode(logits, blank_index):
    """
    Greedily decode the logits using TensorFlow's CTC decoder.
    """
    if logits.dtype != tf.float32:
        logits = tf.cast(logits, tf.float32)
    # Transpose to time-major for the decoder.
    logits_time_major = tf.transpose(logits, [1, 0, 2])
    decoded, _ = tf.nn.ctc_greedy_decoder(
        logits_time_major,
        sequence_length=[logits.shape[1]] * logits.shape[0],
        blank_index=blank_index
    )
    return tf.sparse.to_dense(decoded[0], default_value=-1)

def decode_ids_to_strs(decoded_batch, idx_to_char_map):
    """
    Convert a batch of decoded character IDs to strings.
    """
    batch_strs = []
    for row in decoded_batch.numpy():
        chars = []
        for idx in row:
            if idx == -1:
                break
            if idx > 0:
                ch = idx_to_char_map.get(idx, '?')
                chars.append(ch)
        batch_strs.append("".join(chars))
    return batch_strs

def model_log(msg, log_f):
    print(msg)
    log_f.write(msg + "\n")
    