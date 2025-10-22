import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils, callbacks #type: ignore

tf.random.set_seed(42)

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = utils.to_categorical(y_train, 10), utils.to_categorical(y_test, 10)

# Residual block definition
def residual_block(x, filters, downsample=False):
    shortcut = x
    stride = 1
    if downsample:
        stride = 2
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

# Build a simple ResNet (20 layers)
def build_resnet(input_shape=(32,32,3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Stack residual blocks
    for filters in [16, 32, 64]:
        x = residual_block(x, filters, downsample=(filters != 16))
        x = residual_block(x, filters)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, x, name="SimpleResNet")

model = build_resnet()
model.summary()

# Compile & train
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cb = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
]

history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=20, batch_size=128, callbacks=cb)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
