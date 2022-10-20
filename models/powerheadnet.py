"""
【已废弃】新脚本见handcraft_model
"""
a =
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
from tensorflow.keras import layers

# from tensorflow.python.keras.layers import Conv2D, Input, Add, Multiply, UpSampling2D, Conv2DTranspose, DepthwiseConv2D, \
#     Dropout, MaxPooling2D, concatenate
# from tensorflow.python.keras.models import Model

from tensorflow.keras.layers import Conv2D, Input, Add, Multiply, UpSampling2D, Conv2DTranspose, DepthwiseConv2D, \
    Dropout, MaxPooling2D, concatenate
from tensorflow.keras.models import Model


def powerhead_model_1(img_size=(32, 229)):
    """
    本次使用的模型
    :param img_size:
    :return:
    """
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=img_size + (2,))


    # 编码器
    conv2d = ConvBlock(inputs, 16, kernel_size=3, stride=1, activation=hard_swish, name="L_conv2d")  # 跳层0
    conv2d_1_dw = DWBlock(conv2d, 16, 3, 2, activation=relu, name="L_conv2d_1_dw")
    conv2d_1_att = Multiply(name="L_conv2d_1_att")([conv2d_1_dw, AttBlock(conv2d_1_dw, 8, 16, name="LB_conv2d_1_att")])

    conv2d_4 = ConvBlock(conv2d_1_att, 16, kernel_size=1, name="L_conv2d_4")  # 跳层1
    conv2d_5_dw = DWBlock(conv2d_4, 72, 3, 1, activation=relu, name="L_conv2d_5_dw")
    conv2d_6 = ConvBlock(conv2d_5_dw, 24, kernel_size=1, name="L_conv2d_6")
    conv2d_7_dw = DWBlock(conv2d_6, 88, 3, 1, activation=relu, name="L_conv2d_7_dw")
    conv2d_8 = ConvBlock(conv2d_7_dw, 24, kernel_size=1, name="L_conv2d_8")
    add = Add(name="L_add")([conv2d_6, conv2d_8])  # 跳层2

    conv2d_9_dw = DWBlock(add, 96, 5, 1, activation=hard_swish, name="L_conv2d_9_dw")
    conv2d_9_att = Multiply(name="L_conv2d_9_att")([conv2d_9_dw, AttBlock(conv2d_9_dw, 24, 96, name="LB_conv2d_9_att")])
    outputs = conv2d_9_att


    # # baseline 方案 7层
    # """
    # Total params: 169,729
    # Trainable params: 169,729
    # Non-trainable params: 0
    # 138.68115234375
    # """
    # x = layers.Conv2D(64, 5, **conv_args)(inputs)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(32, 3, **conv_args)(x)
    # outputs = layers.Conv2D(1, 3, activation="linear", padding="same")(x)

    return keras.Model(inputs, outputs)


def powerhead_model(img_size=(32, 229)):
    """
    本次使用的模型
    :param img_size:
    :return:
    """
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=img_size + (2,))

    # # baseline 方案 7层
    # """
    # Total params: 169,729
    # Trainable params: 169,729
    # Non-trainable params: 0
    # 138.68115234375
    # """
    # x = layers.Conv2D(64, 5, **conv_args)(inputs)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(32, 3, **conv_args)(x)
    # outputs = layers.Conv2D(1, 3, activation="linear", padding="same")(x)

    # # 提升方案，多两层 9
    # """
    # Total params: 243,585
    # Trainable params: 243,585
    # Non-trainable params: 0
    # 138.27890014648438
    # """
    # x = layers.Conv2D(64, 5, **conv_args)(inputs)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(32, 3, **conv_args)(x)
    # outputs = layers.Conv2D(1, 3, activation="linear", padding="same")(x)

    # # 提升方案，多四层 11
    # """
    # Total params: 317,441
    # Trainable params: 317,441
    # Non-trainable params: 0
    # 132.19369506835938
    # """
    # x = layers.Conv2D(64, 5, **conv_args)(inputs)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(64, 3, **conv_args)(x)
    # x = layers.Conv2D(32, 3, **conv_args)(x)
    # outputs = layers.Conv2D(1, 3, activation="linear", padding="same")(x)

    # 提升方案，多六层 13
    """
    Total params: 391,297
    Trainable params: 391,297
    Non-trainable params: 0
    127.81877899169922
    """
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    outputs = layers.Conv2D(1, 3, activation="linear", padding="same")(x)

    # # 优化方案
    # """
    # Total params: 57,409
    # Trainable params: 57,409
    # Non-trainable params: 0
    # 144.70619201660156
    # """
    # x = layers.Conv2D(32, 5, **conv_args)(inputs)
    # x = layers.Conv2D(32, 3, **conv_args)(x)
    # x = layers.Conv2D(32, 3, **conv_args)(x)
    # x = layers.Conv2D(32, 3, **conv_args)(x)
    # x = layers.Conv2D(32, 3, **conv_args)(x)
    # x = layers.Conv2D(32, 3, **conv_args)(x)
    # x = layers.Conv2D(32, 3, **conv_args)(x)
    # outputs = layers.Conv2D(1, 3, activation="linear", padding="same")(x)

    return keras.Model(inputs, outputs)


def get_model_old(img_size=(32, 229)):
    """
    不要修改次模型结构
    :param img_size:
    :return:
    """
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    #     inputs = keras.Input(shape=(None, None, channels))
    inputs = keras.Input(shape=img_size + (2,))
    # 方案1
    # 一阶
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)  # val_loss: 90.4540

    # 二阶
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)  # val_loss: 50.4543

    x = layers.Conv2D(32, 3, **conv_args)(x)
    outputs = layers.Conv2D(1, 3, activation="linear", padding="same")(x)

    return keras.Model(inputs, outputs)


def unet_mini(img_size=(32, 224)):
    # val_loss: 77.8673
    inputs = keras.Input(shape=img_size + (2,))

    conv1 = Conv2D(32, (3, 3),
                   activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3),
                   activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3),
                   activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3),
                   activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3),
                   activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3),
                   activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2))(
        conv3), conv2])
    conv4 = Conv2D(64, (3, 3),
                   activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3),
                   activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2))(
        conv4), conv1])
    conv5 = Conv2D(32, (3, 3),
                   activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3),
                   activation='relu', padding='same', name="seg_feats")(conv5)

    # outputs = Conv2D(1, 3, activation="linear", padding="same")(x)

    outputs = Conv2D(1, (1, 1), padding='same')(conv5)

    model = keras.Model(inputs, outputs)
    return model


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([x, hard_sigmoid(x)])


def AttBlock(inputs, filters_1, filters_2, name=""):
    prefix = name
    x = layers.GlobalAveragePooling2D(
        keepdims=True, name=prefix + 'squeeze_excite/AvgPool')(
        inputs)
    x = layers.Conv2D(
        filters_1,
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv')(x)

    if filters_2 != 0:
        # 部分层没有扩张部分
        x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
        x = layers.Conv2D(
            filters_2,
            kernel_size=1,
            padding='same',
            name=prefix + 'squeeze_excite/Conv_1')(
            x)
    x = hard_sigmoid(x)
    # x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def DWBlock(input_x, filters, kernel_size, stride, activation, name="", res=False):
    prefix = name
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'expand')(
        input_x)
    x = layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand/BatchNorm')(
        x)
    conv_x = activation(x)

    # if stride == 2:
    #     x = layers.ZeroPadding2D(
    #         padding=imagenet_utils.correct_pad(x, kernel_size))(
    #         x)

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        # padding='same' if stride == 1 else 'valid',  # todo
        padding='same',
        use_bias=False,
        name=prefix + 'depthwise')(
        conv_x)
    x = layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise/BatchNorm')(
        x)
    x = activation(x)
    if res:
        x = Add(name=prefix + "res")([conv_x, x])
    return x


def ConvBlock(x, filters, kernel_size=3, stride=1, name="", activation=relu):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding='same',
        # padding='same' if stride == 1 else 'valid',
        activation=None,
        name=name)(x)
    x = activation(x)
    return x


def get_model(pretrained_weights=None, input_size=(32, 224, 2)):
    # 注意此处使用resize后的频谱
    #

    # dw卷积以第一个卷积命名
    # att以前方最近卷积命名

    inputs = Input(input_size)
    # resized_input = Resizing(height=32, width=224)(inputs)
    resized_input = inputs
    # 编码器
    conv2d = ConvBlock(resized_input, 16, kernel_size=3, stride=2, activation=hard_swish, name="L_conv2d")  # 跳层0
    conv2d_1_dw = DWBlock(conv2d, 16, 3, 2, activation=relu, name="L_conv2d_1_dw")
    conv2d_1_att = Multiply(name="L_conv2d_1_att")([conv2d_1_dw, AttBlock(conv2d_1_dw, 8, 16, name="LB_conv2d_1_att")])

    conv2d_4 = ConvBlock(conv2d_1_att, 16, kernel_size=1, name="L_conv2d_4")  # 跳层1
    conv2d_5_dw = DWBlock(conv2d_4, 72, 3, 2, activation=relu, name="L_conv2d_5_dw")
    conv2d_6 = ConvBlock(conv2d_5_dw, 24, kernel_size=1, name="L_conv2d_6")
    conv2d_7_dw = DWBlock(conv2d_6, 88, 3, 1, activation=relu, name="L_conv2d_7_dw")
    conv2d_8 = ConvBlock(conv2d_7_dw, 24, kernel_size=1, name="L_conv2d_8")
    add = Add(name="L_add")([conv2d_6, conv2d_8])  # 跳层2

    conv2d_9_dw = DWBlock(add, 96, 5, 2, activation=hard_swish, name="L_conv2d_9_dw")
    conv2d_9_att = Multiply(name="L_conv2d_9_att")([conv2d_9_dw, AttBlock(conv2d_9_dw, 24, 96, name="LB_conv2d_9_att")])

    conv2d_12 = ConvBlock(conv2d_9_att, 32, kernel_size=1, name="L_conv2d_12")
    conv2d_13_dw = DWBlock(conv2d_12, 128, 5, 1, activation=hard_swish, name="L_conv2d_13_dw")
    conv2d_13_att = Multiply(name="L_conv2d_13_att")(
        [conv2d_13_dw, AttBlock(conv2d_13_dw, 32, 128, name="LB_conv2d_13_att")])
    conv2d_16 = ConvBlock(conv2d_13_att, 32, kernel_size=1, name="L_conv2d_16")
    add_1 = Add(name="L_add_1")([conv2d_12, conv2d_16])

    conv2d_17_dw = DWBlock(add_1, 128, 5, 1, activation=hard_swish, name="L_conv2d_17_dw")
    conv2d_17_att = Multiply(name="L_conv2d_17_att")([conv2d_17_dw, AttBlock(conv2d_17_dw, 32, 128)])
    conv2d_20 = ConvBlock(conv2d_17_att, 32, kernel_size=1, name="L_conv2d_20")
    add_2 = Add(name="add_2")([add_1, conv2d_20])

    # 解码器-------------------

    # 16
    conv2d_21_dw = DWBlock(add_2, 96, 5, 1, activation=hard_swish, name="L_conv2d_21_dw")
    conv2d_21_att = Multiply(name="L_conv2d_21_att")(
        [conv2d_21_dw, AttBlock(conv2d_21_dw, 24, 96, name="LB_conv2d_21_att")])
    conv2d_24 = ConvBlock(conv2d_21_att, 32, kernel_size=1, name="L_conv2d_24")
    add_3 = Add(name="L_add_3")([add_2, conv2d_24])

    # 16
    conv2d_25_dw = DWBlock(add_3, 96, 5, 1, activation=hard_swish, name="L_conv2d_25_dw")
    conv2d_25_att = Multiply(name="L_conv2d_25_att")(
        [conv2d_25_dw, AttBlock(conv2d_25_dw, 24, 96, name="LB_conv2d_25_att")])
    conv2d_28 = ConvBlock(conv2d_25_att, 32, kernel_size=1, name="L_conv2d_28")
    add_4 = Add(name="L_add_4")([add_3, conv2d_28])
    # 16
    conv2d_29 = ConvBlock(add_4, 128, kernel_size=1, name="L_conv2d_29")
    conv2d_29_att = Multiply()([conv2d_29, AttBlock(add_4, 128, 0, name="LB_conv2d_29_att")])

    # 16 -> 32
    upsampling = UpSampling2D(size=(2, 2), name="L_upsampling")(conv2d_29_att)
    conv2d_31 = ConvBlock(upsampling, 24, kernel_size=1, name="L_conv2d_31")

    conv2d_31_att = Multiply(name="L_conv2d_31_att")(
        [add, AttBlock(Add(name="LB_conv2d_31_att_add")([conv2d_31, add]), 24, 24, name="L_conv2d_31_att")])  # 跳层2

    # 测试脚本，待删除
    # tmp1 = Add(name="LB_conv2d_31_att_add")([conv2d_31, add])
    # tmp2 = AttBlock(tmp1, 24, 24, name="L_conv2d_31_att")
    # conv2d_31_att = Multiply(name="L_conv2d_31_att")(
    #     [add, tmp2])  # 跳层2

    add_6 = Add(name="L_add_6")([conv2d_31, conv2d_31_att])

    conv2d_34_dw = DWBlock(add_6, 24, 3, 1, activation=relu, res=True, name="L_conv2d_34_dw")
    upsampling_1 = UpSampling2D(size=(2, 2), name="L_upsampling_1")(conv2d_34_dw)
    conv2d_35 = ConvBlock(upsampling_1, 16, kernel_size=1, name="L_conv2d_35")
    conv2d_35_att = Multiply(name="L_conv2d_35_att")([conv2d_4, AttBlock(
        Add(name="LB_L_conv2d_35_att_add")([conv2d_35, conv2d_4]), 16, 16, name="LB_conv2d_35_att")])  # 跳层1
    add_9 = Add(name="L_add_9")([conv2d_35, conv2d_35_att])

    conv2d_38_dw = DWBlock(add_9, 16, 3, 1, activation=relu, res=True, name="L_conv2d_38_dw")
    upsampling_2 = UpSampling2D(size=(2, 2), name="L_upsampling_2")(conv2d_38_dw)
    conv2d_39 = ConvBlock(upsampling_2, 16, kernel_size=1, name="L_conv2d_39")
    # 屏蔽远端跳层++++++++++++++++++++++++

    conv2d_39_att = Multiply(name="L_conv2d_39_att")([conv2d,
                                                      AttBlock(Add(name="LB_conv2d_39_att_add")([conv2d, conv2d_39]),
                                                               16,
                                                               16, name="LB_L_conv2d_39_att")])  # 跳层0
    add_12 = Add(name="L_add_12")([conv2d_39, conv2d_39_att])

    # add_12 = conv2d_39
    # 屏蔽远端跳层++++++++++++++++++++++++

    conv42_dw = DWBlock(add_12, 16, 3, 1, activation=relu, res=True, name="L_conv42_dw")

    output = Conv2DTranspose(
        filters=1,
        kernel_size=2,
        strides=2,
        # dilation_rate=2,
        padding="same",
    )(conv42_dw)

    model = Model(inputs=inputs, outputs=output)
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


if __name__ == '__main__':
    # # light
    # # model = get_model(input_size=(32, 224, 2))
    # # Total params: 110,913
    # # Trainable params: 107,809
    # # Non - trainable params: 3,104
    #
    # # model = get_model_old((32, 229))
    # # Total params: 169,729
    # # Trainable params: 169,729
    # # Non-trainable params: 0
    #
    # # model = unet_mini()
    # model = powerhead_model()
    # model.summary()


    model = powerhead_model_1()
    model.summary()