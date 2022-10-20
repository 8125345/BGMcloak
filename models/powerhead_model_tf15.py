"""
mbv3临时模型
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Conv2D, Input, Add, Multiply, UpSampling2D, Conv2DTranspose, DepthwiseConv2D, \
    Dropout, MaxPooling2D, Concatenate, Lambda, Reshape, LayerNormalization
from tensorflow.keras.models import Model
from keras import backend
from keras.applications import imagenet_utils
from keras import backend


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([x, hard_sigmoid(x)])


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)  # 确保永远是divisor的倍数
    # 避免求整后卷积和数量比预期低10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    # x = layers.GlobalAveragePooling2D(
    #     keepdims=True, name=prefix + 'squeeze_excite/AvgPool')(
    #     inputs)
    x = layers.GlobalAveragePooling2D(
        name=prefix + 'squeeze_excite/AvgPool')(
        inputs)
    x = tf.expand_dims(x, axis=1)
    x = tf.expand_dims(x, axis=2)

    x = layers.Conv2D(
        _depth(filters * se_ratio),
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv')(
        x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv_1')(
        x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride, stride_pos, se_ratio,
                        activation, block_id):
    # stride_pos, 1：单向，只砍边长大的一侧；2:双向
    if stride_pos == 1:
        thin_side_stride = stride if stride == 1 else stride / 2  # 当stride为1的时候不砍半了
        double_stride = (thin_side_stride, stride)
    else:
        double_stride = (stride, stride)

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(
            _depth(infilters * expansion),
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'expand')(
            x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand/BatchNorm')(
            x)
        x = activation(x)
    from keras_applications.mobilenet_v2 import correct_pad
    if stride == 2:
        x = layers.ZeroPadding2D(
            # padding=imagenet_utils.correct_pad(x, kernel_size),
            padding=correct_pad(backend, x, kernel_size),

            name=prefix + 'depthwise/pad')(
            x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=double_stride,
        padding='same' if stride == 1 else 'valid',
        use_bias=False,
        name=prefix + 'depthwise')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise/BatchNorm')(
        x)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'project')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'project/BatchNorm')(
        x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x


def ConvBN(x, filters, kernel_size=3, stride=(1, 1), name="", activation=relu):
    prefix = name
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding='same',
        # padding='same' if stride == 1 else 'valid',
        activation=None,
        name=name)(x)
    x = layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand/BatchNorm')(
        x)
    x = activation(x)
    return x


# Resizing
def powerhead_model_5(img_size=(32, 229), alpha=0.5):
    """
    unet结构测试，注意输入是32*229*2，label是8*229*1，线上模型配套模型
    输入输出使用resize适应维度变化
    更轻的模型结构
    :param alpha:
    :param img_size:
    :return:
    """

    def stack_fn(x_stack_in, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)

        # 下采样
        x = x_stack_in
        x_d1s = x[:, 24:, :, :]  # 8 * 224

        x_d2 = _inverted_res_block(x, 1, depth(16), 3, 2, 2, se_ratio, relu, 0)  # 16 * 112
        x_d2s = x_d2[:, 8:, :, :]  # 8 * 112

        x_d4 = _inverted_res_block(x_d2, 72. / 16, depth(24), 3, 2, 2, None, relu, 1)  # 4.5 8 * 56

        # shape保持
        x_d8 = _inverted_res_block(x_d4, 88. / 24, depth(40), 3, 2, 2, None, relu, 2)  # 3.6  4 * 28
        x_d8 = _inverted_res_block(x_d8, 4, depth(40), kernel, 1, 2, se_ratio, activation, 3)  # 4 * 28

        # 上采样
        x_u4 = UpSampling2D(size=(2, 2), name="x_u4_up")(x_d8)  # 8 * 56
        x_u4 = Concatenate(name="x_u4_add")([x_u4, x_d4])
        x_u4 = _inverted_res_block(x_u4, 4, depth(24), kernel, 1, 2, se_ratio, activation, 4)  # 8 * 56

        x_u2 = UpSampling2D(size=(1, 2), name="x_u2_up")(x_u4)  # 8 * 112
        x_u2 = Concatenate(name="x_u2_add")([x_u2, x_d2s])
        x_u2 = _inverted_res_block(x_u2, 6, depth(24), kernel, 1, 2, se_ratio, activation, 5)  # 8 * 112

        x_u1 = UpSampling2D(size=(1, 2), name="x_u1_up")(x_u2)  # 8 * 224
        x_u1 = Concatenate(name="x_u1_add")([x_u1, x_d1s])

        # x_u1 = Resizing(8, 229)(x_u1)
        x_u1 = tf.image.resize_bilinear(x_u1, (8, 229))

        x_u1 = _inverted_res_block(x_u1, 6, depth(24), kernel, 1, 2, se_ratio, activation, 6)  # 8 * 229
        return x_u1

    alpha = alpha
    kernel = 5
    activation = hard_swish
    se_ratio = 0.25
    input_filter_num = 16

    inputs = keras.Input(shape=img_size + (2,))

    x = inputs
    # 输入增加resize
    # x = Resizing(32, 224)(x)  # 使输入输出一致，避免频繁切片操作
    x = tf.image.resize_bilinear(x, (32, 224))

    x = layers.Conv2D(
        input_filter_num,
        kernel_size=3,
        strides=(1, 1),
        padding='same',
        use_bias=False,
        name='Conv')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999, name='Conv/BatchNorm')(x)
    x = activation(x)
    x_stack = stack_fn(x, kernel, activation, se_ratio)
    x = Conv2D(1, (1, 1), padding='same', name="Predictions")(x_stack)

    def aux_net(x):
        """
        辅助网络，用于flag分类
        :param x:
        :return:
        """
        x = _inverted_res_block(x, 6, 16, kernel, 2, 2, se_ratio, activation, "aux_1")
        x = _inverted_res_block(x, 6, 16, kernel, 2, 2, se_ratio, activation, "aux_2")
        # x = layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = layers.GlobalAveragePooling2D()(x)

        x = tf.expand_dims(x, axis=1)
        x = tf.expand_dims(x, axis=2)
        classes = 1
        x = layers.Conv2D(classes, kernel_size=1, padding='same', name='Logits')(x)
        x = layers.Flatten()(x)
        x = layers.Activation(activation="sigmoid",
                              name='AUX_Predictions')(x)

        return x

    x_aux = aux_net(x_stack)
    model = Model(inputs=inputs, outputs=[x, x_aux], name="powerhead_w_flag")

    return model


if __name__ == '__main__':
    # 线上模型专用
    model = powerhead_model_5(img_size=(32, 229), alpha=0.5)  # Total params: 109,314

    model.summary()
