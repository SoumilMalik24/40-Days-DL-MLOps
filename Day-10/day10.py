import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers #type: ignore
import tensorflow_datasets as tfds #type: ignore
import matplotlib.pyplot as plt

# Load Flowers dataset
dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
train_size = int(0.8 * info.splits["train"].num_examples)
val_size = int(0.1 * info.splits["train"].num_examples)
test_size = int(0.1 * info.splits["train"].num_examples)

# Split dataset
train_ds = dataset["train"].take(train_size)
val_ds = dataset["train"].skip(train_size).take(val_size)
test_ds = dataset["train"].skip(train_size + val_size).take(test_size)

# Preprocessing & Augmentation
IMG_SIZE = 160
BATCH_SIZE = 32

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Load pretrained base
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Build model
model = keras.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(5, activation="softmax")
])

# Compile
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train with EarlyStopping
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[early_stop])

# Evaluate
test_loss, test_acc = model.evaluate(test_ds)
print("✅ Final Test Accuracy:", round(test_acc * 100, 2), "%")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Transfer Learning on Flowers Dataset")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
