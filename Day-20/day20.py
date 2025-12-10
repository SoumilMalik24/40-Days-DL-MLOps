# imports & seed (if needed)
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, preprocessing, callbacks #type:ignore
import numpy as np

tf.random.set_seed(42)

# ---- Load Data ----
vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=vocab_size)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

# ---- Fixed Attention Layer ----
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1)

    def call(self, lstm_output, mask=None):
        score = self.score_dense(lstm_output)          # (batch, timesteps, 1)
        weights = tf.nn.softmax(score, axis=1)        # normalize across timesteps
        context = tf.reduce_sum(weights * lstm_output, axis=1)  # (batch, features)
        return context

# ---- Build Model ----
inputs = layers.Input(shape=(max_len,))
x = layers.Embedding(vocab_size, 128)(inputs)
lstm_out = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
context = AttentionLayer()(lstm_out)
outputs = layers.Dense(1, activation='sigmoid')(context)

model = models.Model(inputs, outputs)
model.summary()

# ---- Compile & Train ----
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cb = [callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=128,
    callbacks=cb
)

loss, acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")
