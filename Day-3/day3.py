import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers #type:ignore
import matplotlib.pyplot as plt

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# Model definition
def build_model():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    return model

optimizers = {
    "SGD": "sgd",
    "SGD+Momentum": keras.optimizers.SGD(momentum=0.9),
    "RMSProp": "rmsprop",
    "Adam": "adam"
}

histories = {}
results = {}

# Train with different optimizers
for name, opt in optimizers.items():
    model = build_model()
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    print(f"\nTraining with {name}")
    history = model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0,
                        validation_data=(x_test, y_test))
    histories[name] = history
    results[name] = model.evaluate(x_test, y_test, verbose=0)[1]
    print(f"Test Accuracy with {name}: {results[name]:.4f}")

# Plot validation accuracy curves
plt.figure(figsize=(10,6))
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=name)

plt.title("Optimizer Comparison on MNIST")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
