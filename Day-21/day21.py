"""
Day 21 — Transformer From Scratch
Mini Transformer Encoder for IMDB sentiment classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models, datasets, preprocessing, callbacks #type: ignore

tf.random.set_seed(42)

# ---- Load Data ----
vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=vocab_size)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

# ---- Positional Encoding ----
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angles = pos / tf.pow(10000.0, (2*(i//2))/d_model)
        pe = tf.where(i % 2 == 0, tf.sin(angles), tf.cos(angles))
        self.pos_encoding = pe[tf.newaxis, ...]

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

# ---- Transformer Encoder Block ----
def transformer_encoder(embed_dim, num_heads, ff_dim):
    inputs = layers.Input(shape=(None, embed_dim))
    
    # Self-attention
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_out)

    # Feed-forward
    ffn = layers.Dense(ff_dim, activation='relu')(x)
    ffn = layers.Dense(embed_dim)(ffn)
    outputs = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    return models.Model(inputs, outputs)

# ---- Build the Transformer Model ----
embed_dim = 64
num_heads = 4
ff_dim = 128

inputs = layers.Input(shape=(max_len,))
x = layers.Embedding(vocab_size, embed_dim)(inputs)
x = PositionalEncoding(max_len, embed_dim)(x)

encoder = transformer_encoder(embed_dim, num_heads, ff_dim)
x = encoder(x)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)
model.summary()

# ---- Train ----
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cb = [callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=8, batch_size=128,
    callbacks=cb
)

loss, acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")
