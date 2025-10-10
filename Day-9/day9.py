import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers #type:ignore

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Resize to 96x96 because MobileNetV2 requires bigger input
x_train = tf.image.resize(x_train, (96, 96))
x_test = tf.image.resize(x_test, (96, 96))

# Load pretrained MobileNetV2 (exclude top classifier)
base_model = keras.applications.MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(96,96,3)
)
base_model.trainable = False  # freeze base layers

# Build transfer learning model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

# Compile
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train
history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                    validation_data=(x_test, y_test))

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Final Test Accuracy:", test_acc)
