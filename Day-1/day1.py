import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load MNIST dataset (60,000 training images, 10,000 test images)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize (scale pixels 0-255 → 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # flatten 28x28 image to 784 vector
    keras.layers.Dense(128, activation='relu'), # hidden layer
    keras.layers.Dense(10, activation='softmax') # output (10 classes)
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot training vs validation accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()
