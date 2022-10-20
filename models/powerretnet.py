"""
声音识别相关代码
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
from tensorflow.keras import layers

# from tensorflow.python.keras.layers import Conv2D, Input, Add, Multiply, UpSampling2D, Conv2DTranspose, DepthwiseConv2D, \
#     Dropout, MaxPooling2D, concatenate
# from tensorflow.python.keras.models import Model

from tensorflow.keras.layers import Conv2D, Input, Add, Multiply, UpSampling2D, Conv2DTranspose, DepthwiseConv2D, \
    Dropout, MaxPooling2D, concatenate, Lambda, Reshape, LayerNormalization
from tensorflow.keras.models import Model
from keras import backend
from keras.applications import imagenet_utils

from keras.activations import softmax


# from keras.applications.mobilenet_v3


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
    x = layers.GlobalAveragePooling2D(
        keepdims=True, name=prefix + 'squeeze_excite/AvgPool')(
        inputs)
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

    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
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



import tensorflow as tf


def transpose(input):
    x = tf.transpose(input, perm=[0, 2, 1])
    return x


def expand_dims1(input):
    x = tf.expand_dims(input, axis=1)
    return x


def expand_dims2(input):
    x = tf.expand_dims(input, axis=-1)
    return x


def matmul(input):
    """input must be a  list"""
    return tf.matmul(input[0], input[1])


def gcnet_layer(inputs):
    x = inputs
    bs, h, w, c = x.get_shape().as_list()
    input_x = x
    input_x = Reshape((-1, c))(input_x)  # [N, H*W, C]
    input_x = Lambda(transpose)(input_x)  # [N,C,H*W]
    input_x = Lambda(expand_dims1)(input_x)

    context_mask = Conv2D(filters=1, kernel_size=(1, 1))(x)
    context_mask = Reshape((-1, 1))(context_mask)
    context_mask = softmax(context_mask, axis=1)  # [N, H*W, 1]
    context_mask = Lambda(transpose)(context_mask)
    context_mask = Lambda(expand_dims2)(context_mask)

    context = Lambda(matmul)([input_x, context_mask])  # [N,1,c,1]
    context = Reshape((1, 1, c))(context)

    context_transform = Conv2D(int(c / 8), (1, 1))(context)
    context_transform = LayerNormalization()(context_transform)
    context_transform = relu(context_transform)
    context_transform = Conv2D(c, (1, 1))(context_transform)

    x = keras.layers.add([x, context_transform])

    return x


# # 原始mobilenetv3
# x = _inverted_res_block(x, 1, depth(16), 3, 2, 1, se_ratio, relu, 0)
# x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, 1, None, relu, 1)
# x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, 2, None, relu, 2)
# x = _inverted_res_block(x, 4, depth(40), kernel, 2, 1, se_ratio, activation, 3)
# x = _inverted_res_block(x, 6, depth(40), kernel, 1, 2, se_ratio, activation, 4)
# x = _inverted_res_block(x, 6, depth(40), kernel, 1, 2, se_ratio, activation, 5)
# x = _inverted_res_block(x, 3, depth(48), kernel, 1, 2, se_ratio, activation, 6)
# x = _inverted_res_block(x, 3, depth(48), kernel, 1, 2, se_ratio, activation, 7)
# x = _inverted_res_block(x, 6, depth(96), kernel, 2, 2, se_ratio, activation, 8)
# x = _inverted_res_block(x, 6, depth(96), kernel, 1, 2, se_ratio, activation, 9)
# x = _inverted_res_block(x, 6, depth(96), kernel, 1, 2, se_ratio, activation, 10)
# from keras.applications.mobilenet_v3 import
def rec_model(img_size=(32, 229), alpha=0.5, model_type="small", last_activation=True, last_point_ch=1280):
    """
    声音识别模型
    :param img_size:
    :param alpha:
    :return:
    """
    assert model_type in ["small", "full"]

    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)

        if model_type == "small":
            print("使用精简版声音识别")
            stride_mode = 2
            # 自定义1 handcraft_20220406_1
            x = _inverted_res_block(x, 1, depth(16), 3, 2, stride_mode, se_ratio, relu, 0)
            x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, stride_mode, None, relu, 1)  # 4.5
            x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, 2, None, relu, 2)  # 3.6
            x = _inverted_res_block(x, 4, depth(40), kernel, 2, 2, se_ratio, activation, 3)
            # x = _inverted_res_block(x, 6,           depth(40), kernel,  1, 2, se_ratio, activation, 4)
            x = _inverted_res_block(x, 6, depth(40), kernel, 1, 2, se_ratio, activation, 5)
            x = _inverted_res_block(x, 3, depth(48), kernel, 1, 2, se_ratio, activation, 6)
            # x = _inverted_res_block(x, 3,           depth(48), kernel,  1, 2, se_ratio, activation, 7)
            x = _inverted_res_block(x, 6, depth(96), kernel, 2, 2, se_ratio, activation, 8)
            x = _inverted_res_block(x, 6, depth(96), kernel, 1, 2, se_ratio, activation, 9)
            # x = _inverted_res_block(x, 6,           depth(96), kernel,  1, 2, se_ratio, activation, 10)
        else:
            # 原始mb3
            print("使用原版声音识别")
            x = _inverted_res_block(x, 1, depth(16), 3, 2, 2, se_ratio, relu, 0)
            x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, 2, None, relu, 1)
            x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, 2, None, relu, 2)
            x = _inverted_res_block(x, 4, depth(40), kernel, 2, 2, se_ratio, activation, 3)
            x = _inverted_res_block(x, 6, depth(40), kernel, 1, 2, se_ratio, activation, 4)
            x = _inverted_res_block(x, 6, depth(40), kernel, 1, 2, se_ratio, activation, 5)
            x = _inverted_res_block(x, 3, depth(48), kernel, 1, 2, se_ratio, activation, 6)
            x = _inverted_res_block(x, 3, depth(48), kernel, 1, 2, se_ratio, activation, 7)
            x = _inverted_res_block(x, 6, depth(96), kernel, 2, 2, se_ratio, activation, 8)
            x = _inverted_res_block(x, 6, depth(96), kernel, 1, 2, se_ratio, activation, 9)
            x = _inverted_res_block(x, 6, depth(96), kernel, 1, 2, se_ratio, activation, 10)

        return x

    alpha = alpha
    kernel = 5
    activation = hard_swish
    se_ratio = 0.25
    dropout_rate = 0.2
    # last_point_ch = last_point_ch  # 128

    inputs = keras.Input(shape=img_size + (1,))

    x = inputs
    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(1, 2),
        padding='same',
        use_bias=False,
        name='Conv')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999, name='Conv/BatchNorm')(x)
    x = activation(x)
    x = stack_fn(x, kernel, activation, se_ratio)

    # GCnet相关代码
    # # x = Lambda(gcnet_layer)(x)
    # bs, h, w, c = x.get_shape().as_list()
    # input_x = x
    # input_x = Reshape((-1, c))(input_x)  # [N, H*W, C]
    # input_x = Lambda(transpose)(input_x)  # [N,C,H*W]
    # input_x = Lambda(expand_dims1)(input_x)
    #
    # context_mask = Conv2D(filters=1, kernel_size=(1, 1))(x)
    # context_mask = Reshape((-1, 1))(context_mask)
    # context_mask = softmax(context_mask, axis=1)  # [N, H*W, 1]
    # context_mask = Lambda(transpose)(context_mask)
    # context_mask = Lambda(expand_dims2)(context_mask)
    #
    # context = Lambda(matmul)([input_x, context_mask])  # [N,1,c,1]
    # context = Reshape((1, 1, c))(context)
    #
    # context_transform = Conv2D(int(c / 8), (1, 1))(context)
    # context_transform = LayerNormalization()(context_transform)
    # context_transform = relu(context_transform)
    # context_transform = Conv2D(c, (1, 1))(context_transform)
    #
    # x = keras.layers.add([x, context_transform])

    # 尾部
    channel_axis = -1

    last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_point_ch = _depth(last_point_ch * alpha)
    x = layers.Conv2D(
        last_conv_ch,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name='Conv_1')(x)
    x = layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999, name='Conv_1/BatchNorm')(x)
    x = activation(x)
    # top
    x = layers.GlobalAveragePooling2D(keepdims=True)(x)
    x = layers.Conv2D(
        last_point_ch,
        kernel_size=1,
        padding='same',
        use_bias=True,
        name='Conv_2')(x)
    x = activation(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    # classes = 88 * 32
    classes = 88 * 2
    x = layers.Conv2D(classes, kernel_size=1, padding='same', name='Logits')(x)

    if last_activation:
        x = layers.Flatten()(x)
        # imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Activation(activation="sigmoid",
                              name='Predictions')(x)
    else:
        x = layers.Flatten(name="Predictions")(x)

    # 直接输出测试[不会使用]
    # x = Conv2D(1, (1, 1), padding='same')(x)
    model = Model(inputs, x, name=f"rec_{alpha}")

    return model


if __name__ == '__main__':
    model = rec_model(img_size=(32, 229), model_type="full", alpha=1.0, last_point_ch=640)
    model.summary()

    # model = powerhead_model_3(img_size=(32, 224), alpha=0.5)
    # model.summary()
"""
last_point_ch = 1280

small 0.5
761,752

small 1.0 
1,517,480

full 1.0
1,902,848

full 2.0 20M
Total params: 6,964,352


last_point_ch = 640

small 0.5
464,152

small 1.0
1,035,560

full 1.0
1,420,928

full 2.0
5,263,232
"""