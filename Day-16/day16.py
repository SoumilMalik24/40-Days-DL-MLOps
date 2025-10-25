"""
Day 16 — RNN Fundamentals
Simple RNN for sentiment analysis on IMDB dataset using Keras.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, datasets, preprocessing, utils, callbacks #type: ignore

tf.random.set_seed(42)

# --- Load IMDB dataset ---
vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=vocab_size)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

# --- Build RNN model ---
model = models.Sequential([
    layers.Embedding(vocab_size, 64, input_length=max_len),
    layers.SimpleRNN(64, activation='tanh', return_sequences=False),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

# --- Compile & Train ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cb = [callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]

history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=10, batch_size=128,
                    callbacks=cb)

# --- Evaluate ---
loss, acc = model.evaluate(x_test, y_test)
print(f"nTest Accuracy: {acc*100:.2f}%")