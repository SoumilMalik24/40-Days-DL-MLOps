import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils, callbacks

tf.random.set_seed(42)

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = utils.to_categorical(y_train, 10), utils.to_categorical(y_test, 10)

# Define VGG block
def vgg_block(x, filters, conv_layers):
    for _ in range(conv_layers):
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    return x

# Build VGG16-like model
def build_vgg16(input_shape=(32,32,3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = vgg_block(inputs, 64, 2)
    x = vgg_block(x, 128, 2)
    x = vgg_block(x, 256, 3)
    x = vgg_block(x, 512, 3)
    x = vgg_block(x, 512, 3)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name='VGG16_Simplified')

model = build_vgg16()
model.summary()

# Compile & Train
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cb = [callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)]

history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=30, batch_size=128,
                    callbacks=cb)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n Test Accuracy: {test_acc*100:.2f}")