import tensorflow as tf
from tensorflow.keras import layers

def build_crnn_model(img_height=128, img_width=800, vocab_size=80):
    """
    A CRNN with four conv blocks and dropout in the BiLSTM layers.
    """
    input_img = layers.Input(shape=(img_height, img_width, 1), name='image_input', dtype='float32')

    # Convolution block 1
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)  # width / 2

    # Convolution block 2
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)  # width / 4 overall

    # Convolution block 3
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)  # width / 8 overall
    
    # Convolution block 4
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)  # width / 16 overall

    # Reshape (batch, h, w, channels) => (batch, w, h*channels(features) )
    shape = x.shape
    bsz, new_h, new_w, filters = shape[0], shape[1], shape[2], shape[3]
    x = layers.Reshape((new_w, new_h * filters))(x)

    # BiLSTMs block 1
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    # BiLSTMs block 2
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    # Final Dense => raw logits (no activation)
    logits = layers.Dense(vocab_size, activation='linear')(x)

    model = tf.keras.Model(inputs=input_img, outputs=logits, name='crnn_model_v4')
    return model

@tf.function
def ctc_lambda_func(y_true, y_pred, vocab_size):
    y_pred_32 = tf.cast(y_pred, tf.float32)  # the raw logits
    # label length = number of non-zero chars
    label_length = tf.reduce_sum(tf.cast(y_true != 0, tf.int32), axis=1)
    # input length = number of timesteps in y_pred
    input_length = tf.fill([tf.shape(y_pred_32)[0]], tf.shape(y_pred_32)[1])

    # Pass y_pred_32 as logits
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred_32,
        label_length=label_length,
        logit_length=input_length,
        blank_index=vocab_size - 1,
        logits_time_major=False
    )
    return tf.reduce_mean(loss)
