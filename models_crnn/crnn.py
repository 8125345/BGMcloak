"""
按照线上模式进行训练
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# from tensorflow.python.keras.layers import Conv2D, Input, Add, Multiply, UpSampling2D, Conv2DTranspose, DepthwiseConv2D, \
#     Dropout, MaxPooling2D, concatenate
# from tensorflow.python.keras.models import Model

from tensorflow.keras.layers import Conv2D, Input, Add, Multiply, UpSampling2D, Conv2DTranspose, DepthwiseConv2D, \
    Dropout, MaxPooling2D, concatenate, Lambda, Reshape, LayerNormalization, Resizing
from tensorflow.keras.models import Model
from keras import backend
from keras.applications import imagenet_utils


def cnn_encoder(x, input_shape):
    image_height, image_width = input_shape

    x = keras.layers.Conv2D(
        48,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="conv0",
    )(x)

    x = keras.layers.Conv2D(
        48,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="conv1",
    )(x)
    x = keras.layers.MaxPooling2D((1, 2), name="pool1")(x)
    x = keras.layers.Dropout(0.15)(x)

    last_channel = 96
    x = keras.layers.Conv2D(
        last_channel,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)  # 64
    x = keras.layers.MaxPooling2D((1, 2), name="pool2")(x)
    x = keras.layers.Dropout(0.5)(x)

    new_shape = ((image_height), (image_width // 4) * last_channel)
    x = keras.layers.Reshape(target_shape=new_shape, name="flatten_end")(x)
    x = keras.layers.Dense(768, activation="relu", name="dense1")(x)  # 64
    x = keras.layers.Dropout(0.2)(x)
    return x


def bilstm_decoder(x, units=256):
    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(units, return_sequences=True, dropout=0)  # 128
    )(x)

    return x


def GOOGLE_CRNN(input_shape):

    image_height = input_shape[0]
    image_width = input_shape[1]
    inputs = keras.Input(shape=(image_height, image_width, 1), name="image")

    # 编码器
    x = cnn_encoder(inputs, (image_height, image_width))
    # 解码器
    x = bilstm_decoder(x, units=256)

    x = keras.layers.Dense(
        88, activation="sigmoid", name="dense2"
    )(x)

    # Define the model.
    model = keras.models.Model(
        inputs=inputs, outputs=x, name=f"GOOGLE_CRNN")

    return model


if __name__ == '__main__':
    model = GOOGLE_CRNN(input_shape=(8, 229, 1))  # 宽模型 o8 5.3M Total params: 6,315,928
    model.summary()
