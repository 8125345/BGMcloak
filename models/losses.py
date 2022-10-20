from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

import numpy as np
import tensorflow as tf

from config import image_shape
from config import rec_model_path

mae_loss_fn = tf.keras.losses.mean_absolute_error

# vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
# loss_block3 = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
# loss_block3.trainable = False
#
# loss_block2 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv2').output)
# loss_block2.trainable = False
#
# loss_block1 = Model(inputs=vgg.input, outputs=vgg.get_layer('block1_conv2').output)
# loss_block1.trainable = False
#
#
# def perceptual_loss(img_true, img_generated):
#     return K.mean(K.square(loss_block1(img_true) - loss_block1(img_generated))) + 2 * K.mean(
#         K.square(loss_block2(img_true) - loss_block2(img_generated))) + 5 * K.mean(
#         K.square(loss_block3(img_true) - loss_block3(img_generated)))
#

path = "/deepiano_data/dataset/tflite-models/train-2022-03-16/4990-8/saved_model_bs/1648093312"
# path = '/data/projects/BGMcloak/model_files/export_model/saved_model/1647947972'
new_model = tf.saved_model.load(path, tags="serve")
# /data/projects/BGMcloak/model_files/export_model/saved_model/1647947972
infer = new_model.signatures["serving_default"]
print(infer.structured_outputs)


def rec_loss_fn(y_pred):
    # input_data = tf.reshape(y_pred, [-1])[: 1832]
    b = tf.shape(y_pred)[0]
    input_data = tf.reshape(y_pred, [b, 1832])
    print("rec_loss的shape", input_data.shape)
    labeling = infer(input_data)["onset_probs_flat"]
    # tf.print("rec", labeling)

    topk1 = tf.math.top_k(labeling, k=3)
    tf.print("top10", topk1)
    topk = labeling
    reduce_v = tf.reduce_mean(topk)
    return reduce_v
    # return tf.square(reduce_v)
    # y = reduce_v
    # # x = ln(y / (1 - y))
    # return tf.math.log(y / (1 - y))


def image_to_picies(imgs, tile_size=(32, 32)):
    """

    :param imgs:
    :param tile_size: H*W
    :return:
    """
    #     b, h, w, c = tf.shape(imgs)
    b = tf.shape(imgs)[0]
    h = tf.shape(imgs)[1]
    w = tf.shape(imgs)[2]
    c = tf.shape(imgs)[3]

    # tf.print("图像维度", b, h, w, c)

    #     print(tf.shape(imgs)[0])

    #     b, h, w, c = imgs.get_shape()

    image_shape = (h, w, c)
    #     print(image3)
    #     print(image_shape)

    tile_rows = tf.reshape(imgs, [b, image_shape[0], -1, tile_size[1], image_shape[2]])
    #     print(tile_rows)
    #     print(tile_rows.shape)

    serial_tiles = tf.transpose(tile_rows, [0, 2, 1, 3, 4])
    #     print(serial_tiles)
    #     print(serial_tiles.shape)

    ans = tf.reshape(serial_tiles, [b, -1, tile_size[0], tile_size[1], image_shape[2]])
    #     print(ans)
    #     print(ans.shape)

    ans = tf.reshape(ans, [-1, tile_size[0], tile_size[1], image_shape[2]])
    #     print(ans.shape)
    return ans


def cloak_loss(y_true, y_pred):
    print("y_pred的shape", y_pred.shape)

    mae_loss = mae_loss_fn(y_true, y_pred)
    #     print("mae_loss", mae_loss.shape)
    mae_loss = tf.reduce_mean(mae_loss, axis=None)
    #     print("mae_loss_mean", mae_loss.shape)
    mae_loss = tf.cast(mae_loss, tf.float32)

    new_y_pred = image_to_picies(y_pred, tile_size=(8, 229))  # tile_size=(32, 32)
    print("new_y_pred的shape", new_y_pred.shape)
    # tf.print("切片batch", tf.shape(new_y_pred)[0])

    # p_loss = perceptual_loss(y_true, y_pred)

    weight = 1
    rec_loss = rec_loss_fn(new_y_pred) * weight

    # tf.print("\tmae_loss", mae_loss)
    # # tf.print("\tperceptual_loss", p_loss)
    # tf.print("\trec_loss", rec_loss)

    losses = mae_loss + rec_loss
    # losses = p_loss + rec_loss
    losses = rec_loss
    return losses

def powerhead_loss(y_true, y_pred):
    pass