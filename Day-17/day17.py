"""
Day 17 — LSTM & GRU
Comparison of LSTM and GRU on IMDB sentiment classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, datasets, preprocessing, callbacks #type: ignore

tf.random.set_seed(42)

# --- Load and preprocess IMDB dataset ---
vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=vocab_size)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

# --- LSTM Model ---
def build_lstm():
    model = models.Sequential([
        layers.Embedding(vocab_size, 128, input_length=max_len),
        layers.LSTM(128, return_sequences=False),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# --- GRU Model ---
def build_gru():
    model = models.Sequential([
        layers.Embedding(vocab_size, 128, input_length=max_len),
        layers.GRU(128, return_sequences=False),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# --- Train and evaluate models ---
def train_model(model, name):
    print(f"\\nTraining {name} model...")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cb = [callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=10, batch_size=128, callbacks=cb)
    loss, acc = model.evaluate(x_test, y_test)
    print(f"{name} Test Accuracy: {acc*100:.2f}%\n")
    return history

lstm_model = build_lstm()
gru_model = build_gru()

hist_lstm = train_model(lstm_model, 'LSTM')
hist_gru = train_model(gru_model, 'GRU')
