import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers #type:ignore
import matplotlib.pyplot as plt

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# Build model
def build_model():
    return keras.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])

model = build_model()

# Step 1: Define Early Stopping BEFORE training
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

# Step 2: Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Step 3: Train with Early Stopping
history = model.fit(x_train, y_train, epochs=30, batch_size=64,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stop])

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Final Test Accuracy:", test_acc)

# Plot accuracy
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Effect of Regularization + Early Stopping")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
