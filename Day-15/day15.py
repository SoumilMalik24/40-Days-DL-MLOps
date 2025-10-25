"""
Day 15 — Vision Transformers (ViT)
Minimal ViT implementation for CIFAR-10 using TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils, callbacks #type: ignore

tf.random.set_seed(42)

# --- Hyperparameters ---
PATCH_SIZE = 4
EMBED_DIM = 64
NUM_HEADS = 4
MLP_DIM = 128
NUM_LAYERS = 6
NUM_CLASSES = 10

# --- Load CIFAR-10 ---
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = utils.to_categorical(y_train, 10), utils.to_categorical(y_test, 10)

# --- Patch embedding ---
def create_patches(images, patch_size):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID"
    )
    patch_dim = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dim])
    return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.projection = layers.Dense(embed_dim)
        self.pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        encoded = self.projection(x) + self.pos_embedding(positions)
        return encoded

# --- Build ViT model ---
def build_vit(input_shape=(32,32,3), patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
              num_heads=NUM_HEADS, mlp_dim=MLP_DIM, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES):
    
    inputs = layers.Input(shape=input_shape)
    num_patches = (input_shape[0] // patch_size) ** 2
    
    patches = layers.Lambda(lambda x: create_patches(x, patch_size))(inputs)
    encoded_patches = PatchEncoder(num_patches, embed_dim)(patches)

    for _ in range(num_layers):
        # Layer normalization + attention
        x1 = layers.LayerNormalization()(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])

        # MLP block
        x3 = layers.LayerNormalization()(x2)
        mlp_output = layers.Dense(mlp_dim, activation='gelu')(x3)
        mlp_output = layers.Dense(embed_dim)(mlp_output)
        encoded_patches = layers.Add()([mlp_output, x2])

    # Classification head
    representation = layers.LayerNormalization()(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    outputs = layers.Dense(num_classes, activation='softmax')(representation)

    return models.Model(inputs, outputs, name="MiniViT")

# Compile & Train
model = build_vit()
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cb = [callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)]

history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=30, batch_size=128,
                    callbacks=cb)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
