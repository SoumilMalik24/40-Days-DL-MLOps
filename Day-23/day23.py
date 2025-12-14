"""
Day 23 — Image Captioning
CNN (ResNet50) + LSTM Decoder (conceptual demo)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications

# Image encoder
cnn = applications.ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)
cnn.trainable = False

# Image feature input
image_input = layers.Input(shape=(2048,))
img_embed = layers.Dense(256, activation='relu')(image_input)

# Caption input
caption_input = layers.Input(shape=(20,))
word_embed = layers.Embedding(5000, 256)(caption_input)

# LSTM decoder
x = layers.LSTM(256)(word_embed)
x = layers.Add()([x, img_embed])
outputs = layers.Dense(5000, activation='softmax')(x)

model = models.Model([image_input, caption_input], outputs)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy'
)

model.summary()
