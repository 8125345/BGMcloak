"""
用于校验json数据正确性
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import tensorflow as tf
import numpy as np

json_path = "/deepiano_data/yuxiaofei/work/data_0718/serialize_0/ai-tagging_train.json"
# 数据生成完成
print("数据加载中...")
with open(json_path, "r") as f:
    data = json.load(f)

# 转为list
data_list = list()
for idx, s_path in data.items():
    data_list.append(s_path)

for idx, path in enumerate(data_list):
    # path = '/deepiano_data/yuxiaofei/work/data_0718/serialize/noise_trans/ai-tagging_20220721_184327/original/2644-车尔尼钢琴初步教程作品 599-599 No.14-3696995/000000.serialized'
    serialized = tf.io.read_file(path)
    # print(path)
    data = tf.io.parse_tensor(serialized, tf.float32)

    # 制作假数据，验证



    feature = data[:, : 293120]
    midi = data[:, 293120:]
    feature = tf.reshape(feature, (640, 229, 2))
    midi = tf.reshape(midi, (640, 88))
    feature = feature.numpy()
    bgm = feature[:, :, 0]
    mix = feature[:, :, 1]
    print(np.array_equal(bgm, mix))

    midi = midi.numpy()
    print(np.any(midi))

    a = 1
    break