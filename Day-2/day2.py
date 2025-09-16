import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define Sigmoid model
sigmoid_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(10, activation='softmax')
])
sigmoid_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

# Define ReLU model
relu_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
relu_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Train both models
history_sigmoid = sigmoid_model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=2)
history_relu = relu_model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=2)

# Evaluate both models on test set
sigmoid_loss, sigmoid_acc = sigmoid_model.evaluate(x_test, y_test)
relu_loss, relu_acc = relu_model.evaluate(x_test, y_test)

print(f"Sigmoid Test Accuracy: {sigmoid_acc:.4f}")
print(f"ReLU Test Accuracy: {relu_acc:.4f}")

# Plot validation accuracy comparison
plt.plot(history_sigmoid.history['val_accuracy'], label='Sigmoid Val Acc')
plt.plot(history_relu.history['val_accuracy'], label='ReLU Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()
