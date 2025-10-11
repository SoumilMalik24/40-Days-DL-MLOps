import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers #type:ignore
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# Shallow CNN
def shallow_cnn():
    return keras.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

# Deep CNN
def deep_cnn():
    return keras.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(32,32,3)),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])

# Compile and train both models
models = {"Shallow": shallow_cnn(), "Deep": deep_cnn()}
history = {}

for name, model in models.items():
    print(f"\nTraining {name} CNN...")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history[name] = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test), verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name} Test Accuracy: {acc*100:.2f}%")

# Plot comparison
plt.figure(figsize=(8,5))
for name in models.keys():
    plt.plot(history[name].history['val_accuracy'], label=f"{name} CNN")
plt.title("Shallow vs Deep CNN Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
