"""
Day 13 — Inception & GoogLeNet
Simplified Inception-v1 architecture on CIFAR-10 using Keras.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils, callbacks #type: ignore

tf.random.set_seed(42)

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = utils.to_categorical(y_train, 10), utils.to_categorical(y_test, 10)

# Define Inception block
def inception_block(x, f1, f3_in, f3, f5_in, f5, pool_proj):
    path1 = layers.Conv2D(f1, 1, padding='same', activation='relu')(x)
    
    path2 = layers.Conv2D(f3_in, 1, padding='same', activation='relu')(x)
    path2 = layers.Conv2D(f3, 3, padding='same', activation='relu')(path2)
    
    path3 = layers.Conv2D(f5_in, 1, padding='same', activation='relu')(x)
    path3 = layers.Conv2D(f5, 5, padding='same', activation='relu')(path3)
    
    path4 = layers.MaxPooling2D(3, strides=1, padding='same')(x)
    path4 = layers.Conv2D(pool_proj, 1, padding='same', activation='relu')(path4)
    
    return layers.concatenate([path1, path2, path3, path4], axis=-1)

# Build simplified GoogLeNet
def build_inception(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)
    
    x = inception_block(x, 32, 48, 64, 8, 16, 16)
    x = inception_block(x, 64, 64, 96, 16, 32, 32)
    x = layers.MaxPooling2D(2)(x)
    x = inception_block(x, 64, 96, 128, 16, 32, 32)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name="SimpleInceptionNet")

model = build_inception()
model.summary()

# Compile and train
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cb = [callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)]

history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=25, batch_size=128,
                    callbacks=cb)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
