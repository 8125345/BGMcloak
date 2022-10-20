"""
测试tf.data流水线中
"""
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

batch_size = 3
frame_num = 8  # 长序列总长度
chunk_size = 2  # 实际使用的序列长度
chunk_in_frame_num = int((frame_num / chunk_size))
n_mels = 5

label_frame_num = 8
label_key_num = 3

# 单张图像级别
test_data = np.arange(batch_size * frame_num * n_mels).reshape((batch_size, frame_num, n_mels))  # 原始数据维度640 * 229
# test_label = test_data + 1000


test_label = np.arange(batch_size * label_frame_num * label_key_num).reshape(
    (batch_size, label_frame_num, label_key_num)) + 1000
rng = tf.random.Generator.from_seed(123, alg='philox')


# test_data = tf.constant(test_data)
# test_label = tf.constant(test_label)
# print(test_data.shape, test_data)


def split(data):
    feature = data[: n_mels]
    label = data[n_mels:]

    # feature_list = list()
    # label_list = list()

    # for i in range()
    return tf.data.Dataset.from_tensor_slices({
        "f": [
            feature[0],
            feature[1]
        ],
        "l": [
            label[0],
            label[1]
        ],
    })


def load_and_aug(data):
    seed = rng.make_seeds(2)[0]
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

    feature = data["feature"]
    label = data["label"]
    print(feature.shape, label.shape)

    concat = tf.concat([feature, label], axis=1)
    print(concat.shape)

    # random_crop = tf.image.stateless_random_crop(value=concat, size=())
    random_crop = tf.image.stateless_random_crop(value=concat, size=(frame_num - chunk_size, n_mels + label_key_num),
                                                 seed=new_seed)

    chunks = tf.reshape(random_crop, (chunk_in_frame_num-1, chunk_size, n_mels + label_key_num))
    print("单条数据batch化后维度", chunks.shape)
    return chunks


# # 设计假数据对
# dataset_list = list()
# for i in range(batch_size):
#     # dataset_list.append([test_data[i], test_label[i]])  # 在非对齐数据中会报错
# print("原始输入", dataset_list)
# print(dataset_list[0])

# 一、数据加载
# 矩阵还原
# 按照时间维度进行拼接
# 随机裁剪
# 传给下级

# 二、数据转为batch
# 数据重组为输入格式
# flat_map

# 三、添加字典key等
dataset_dict = dict()
dataset_dict["feature"] = list()
dataset_dict["label"] = list()

for i in range(batch_size):
    dataset_dict["feature"].append(test_data[i])
    dataset_dict["label"].append(test_label[i])
print("原始输入", dataset_dict)

# 使用tf.data读取数据
dataset = tf.data.Dataset.from_tensor_slices(dataset_dict)
# print("输入抽检", list(dataset.as_numpy_iterator())[0])
dataset = dataset.map(load_and_aug)
print("增强结果", list(dataset.as_numpy_iterator()))
# print("增强抽检", list(dataset.as_numpy_iterator())[0])

dataset = dataset.unbatch()
print("unbatch结果")
for item in list(dataset.as_numpy_iterator()):
    print(item)

# # 单条数据拆分为多条数据
# dataset = dataset.flat_map(split)

# print("切分抽检")
# print(list(dataset.as_numpy_iterator()))

# # --------------------------------------
# # 上述方法有效，再测试unbatch方法
#
# dataset_dict = dict()
# dataset_dict["feature"] = list()
# dataset_dict["label"] = list()
#
# for i in range(batch_size):
#     dataset_dict["feature"].append(test_data[i])
#     dataset_dict["label"].append(test_label[i])
# print("原始输入", dataset_dict)
#
#
# def some_patches_map_func(data):
#     # return tf.stack([
#     #     image[10 : 10 + 256, 20 : 20 + 256],
#     #     image[100 : 100 + 256, 100 : 100 + 256],
#     #     image[500 : 500 + 256, 200 : 200 + 256],
#     # ])
#     feature = data["feature"]
#     label = data["label"]
#     return feature, label
#
#
# # 使用tf.data读取数据
# dataset = tf.data.Dataset.from_tensor_slices(dataset_dict)
# dataset = dataset.map(some_patches_map_func)
# dataset = dataset.unbatch()
# print("切分抽检")
#
# rst = list(dataset.as_numpy_iterator())
# for idx, item in enumerate(rst):
#     print(idx)
#     print(item)
