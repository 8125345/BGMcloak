"""
用于新数据集切片测试
将batch个长序列调整为batch * n * num_short
"""

import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# # 原来
# # 先测试numpy版本
# batch_size = 3
# frame_num = 8  # 长序列总长度
# chunk_size = 2  # 实际使用的序列长度
# chunk_in_frame_num = int((frame_num / chunk_size))
# n_mels = 5
# test_data = np.arange(batch_size * frame_num * n_mels).reshape((batch_size, frame_num, n_mels))  # 原始数据维度640 * 229
#
# print("长频谱", test_data.shape, test_data)
# print("batch size 0", test_data[0])
# print("=" * 100)
#
# # 转化为短频谱
# short_specs = test_data.reshape((batch_size, int((frame_num / chunk_size)), chunk_size, n_mels))
# print("长频谱切分", short_specs.shape, short_specs)
# print("-" * 100)
#
# # 合并batch维度信息
# short_specs = np.concatenate(short_specs, axis=0)
# print(short_specs, short_specs.shape)
#
# print(short_specs[0])
#
# print("*" * 200)
# ----------------------------------------------


#
# batch_size = 3
# frame_num = 8  # 长序列总长度
# chunk_size = 2  # 实际使用的序列长度
# chunk_in_frame_num = int((frame_num / chunk_size))
# n_mels = 5

# batch级别
# test_data = np.arange(batch_size * frame_num * n_mels).reshape((batch_size, frame_num, n_mels))  # 原始数据维度640 * 229


# tf_test_data = tf.constant(test_data)
# short_specs = tf.reshape(tf_test_data, (batch_size, chunk_in_frame_num, chunk_size, n_mels))
#
# print(short_specs.shape)
# print(short_specs)
#
# # 整合chuck到Batch
# # short_specs = tf.stack(short_specs, axis=0)
# short_specs = tf.reshape(short_specs, (batch_size * chunk_in_frame_num, chunk_size, n_mels))
# print(short_specs.shape)
# print(short_specs, short_specs[0])
# ----------------------------------------------
# 调试batch随机切分
batch_size = 3
frame_num = 8  # 长序列总长度
chunk_size = 2  # 实际使用的序列长度
chunk_in_frame_num = int((frame_num / chunk_size))
n_mels = 5

# 单张图像级别
test_data = np.arange(frame_num * n_mels).reshape((frame_num, n_mels))  # 原始数据维度640 * 229
test_data = tf.constant(test_data)
print(test_data.shape, test_data)

crop_data = tf.image.random_crop(value=test_data, size=(frame_num - 2, n_mels))

print(crop_data)

# ----------------------------------------------

# # 函数调试
#
# def data_to_batch(data):
#     """
#     将单独数据整合为batch
#     :param data:
#     :return:
#     """
