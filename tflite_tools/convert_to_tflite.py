import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
# 加载模型
from tensorflow.keras.models import load_model
from models.model_concatenater import concat_model

# model = load_model("/data/projects/BGMcloak/model_files/CNN_20220402_1.h5")

# model = powerhead_model_1(img_size=(32, 229), alpha=0.5)

# 声音识别
from models.handcraft_model import rec_model
model = rec_model(img_size=(32, 229), model_type="full", alpha=1.0)
# model_root_folder = "/data/projects/BGMcloak/models_files"
# rec_model_path = os.path.join(model_root_folder, "rec_20220415_1.h5")
# model = load_model(rec_model_path)
model_name = 'rec_full_1.tflite'

# # 224背景音
# from models.handcraft_model import powerhead_model_1, powerhead_model_2, powerhead_model_3
# model = powerhead_model_3(img_size=(32, 224), alpha=0.5)
# # powerhead_model_2模型需要去掉辅助部分
# from tensorflow.keras.models import Model
# model = Model(model.inputs, model.outputs[0])  # 去掉辅助分支
# model_name = 'powerhead3_05.tflite'

# # 拼接模型
# model_root_folder = "/data/projects/BGMcloak/models_files"
# powerhead_model_path = os.path.join(model_root_folder, "handcraft_20220407_1.h5")
# rec_model_path = os.path.join(model_root_folder, "rec_20220407_1.h5")
# model = concat_model(powerhead_model_path, rec_model_path)


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
save_root = "/data/projects/BGMcloak/model_files"
# model_name = 'powerhead_model_2.tflite'
# model_name = 'rec_small_05.tflite'
# /data/projects/BGMcloak/model_files/rec_small_05.tflite
save_path = os.path.join(save_root, model_name)
print(save_path)
print(os.path.exists(save_path))
with open(save_path, 'wb') as f:
    f.write(tflite_model)

# # 量化模型测试
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
#
# # Save the model.
# with open('my_mobileunet_256_fp16.tflite', 'wb') as f:
#     f.write(tflite_model)
