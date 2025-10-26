"""
Day 19 — Sentiment Analysis Project
Full deep learning pipeline for IMDB sentiment classification using LSTM.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, datasets, preprocessing, callbacks #type: ignore
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

tf.random.set_seed(42)

# --- Load and preprocess data ---
vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=vocab_size)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

# --- Build LSTM model ---
model = models.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_len),
    layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

# --- Compile and Train ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cb = [callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]

history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=10, batch_size=128,
                    callbacks=cb)

# --- Evaluate ---
loss, acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# --- Visualize training ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')
plt.show()

# --- Confusion Matrix ---
y_pred = (model.predict(x_test) > 0.5).astype('int32')
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.show()

# --- Classification Report ---
print(classification_report(y_test, y_pred))

# --- Save model ---
model.save('sentiment_lstm_model.h5')
print("Model saved as sentiment_lstm_model.h5")
