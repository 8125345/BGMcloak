"""
模型转tflite脚本
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 下面几行代码确保转换不报错
import sys

bin_path = os.path.dirname(sys.executable)

if 'PATH' in os.environ:
    os.environ['PATH'] += ':' + bin_path
else:
    os.environ['PATH'] = bin_path

import tensorflow as tf
from tensorflow.keras.models import Model


def convert2lite(model, save_path):
    print(save_path)
    # assert os.path.exists(save_path)
    # converter = tf.compat.v2.lite.TFLiteConverter.from_keras_model(model)
    converter = tf.lite.TFLiteConverter.from_keras_model_file(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 添加后推理速度显著变慢

    # 线上模型配置
    # converter.post_training_quantize = True
    # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_model = converter.convert()
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    print("保存完成")


def convert2lite_lstm(model, save_path):
    print(save_path)
    run_model = tf.function(lambda x: model(x))
    # This is important, let's fix the input size.
    BATCH_SIZE = 1
    STEPS = 8
    INPUT_SIZE = 224
    # concrete_func = run_model.get_concrete_function(
    #     tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE * STEPS * INPUT_SIZE], model.inputs[0].dtype))

    # model directory.
    MODEL_DIR = os.path.split(save_path)[0]
    model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    print("保存完成")


if __name__ == '__main__':
    # 背景音模型
    # # 测试
    # from tensorflow.keras.models import load_model
    #
    # model_name = 'tf15_PowerRec_CRNN_20220526_2.tflite'
    # model_root_folder = "/data/projects/BGMcloak/model_files"
    # # 空模型
    # # model = PowerRec_CRNN(input_shape=(8, 224, 1), decoder_dim=256) # 256 6M
    # # rec_model_path = os.path.join("/data/projects/BGMcloak/train_output_models/powerrec_20220526_2_46_0.00531.h5")
    # # model = load_model(rec_model_path, compile=False)
    #
    # save_path = os.path.join(model_root_folder, model_name)
    # MODEL_DIR = os.path.split(save_path)[0]
    # converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    # # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #
    # tflite_model = converter.convert()
    # with open(save_path, 'wb') as f:
    #     f.write(tflite_model)
    # print("保存完成")
    # ========================================

    # # 背景音模型
    # from models_powerhead_tf15.powerhead_model import PowerHead
    # model = PowerHead(input_shape=(32, 229, 2), alpha=0.5)  # Total params: 100,208
    # model_name = 'powerhead_tf15.tflite'
    # model_root_folder = "/data/projects/BGMcloak/model_files"
    #
    # model.summary()
    # keras_model_path = os.path.join(model_root_folder, "tmp.h5")
    # tf.keras.models.save_model(
    #     model, keras_model_path
    # )
    # convert2lite(keras_model_path, os.path.join(model_root_folder, model_name))

    # ========================================
    # 声音识别模型
    # tf.enable_resource_variables()
    from models_powerrec_tf15.powerrec import PowerRec

    model_name = 'powerrec_tf15.tflite'
    model_root_folder = "/data/projects/BGMcloak/model_files"

    # model = PowerRec(input_shape=(8, 229, 1), alpha=0.8, channel_num=96,
    #                  output_range=None)
    # model = tf.keras.models.load_model("/data1/projects/BGMcloak/train_output_models/powerrec_20220608_1_13_0.17246.h5",
    #                                    compile=False)
    model_path = "/data/projects/BGMcloak/train_output_models/powerrec_20220609_1_40_0.14080.h5"
    model = tf.keras.models.load_model(model_path, compile=False)

    model.summary()
    keras_model_path = os.path.join(model_root_folder, "tmp.h5")
    tf.keras.models.save_model(
        model, keras_model_path
    )
    convert2lite(keras_model_path, os.path.join(model_root_folder, model_name))

    # # ========================================
    # # 拼接模型
    # from models_powerhead_tf15.powerhead_model import PowerHead
    # from models_powerrec_tf15.powerrec import PowerRec
    #
    # powerhead_model = PowerHead(input_shape=(32, 229, 2), alpha=0.6)
    # powerrec_model = PowerRec(input_shape=(8, 229, 1), alpha=0.8, channel_num=96,
    #                  output_range=None)
    #
    # model_name = 'concat_tf15.tflite'
    # model_root_folder = "/data/projects/BGMcloak/model_files"
    #
    # new_rec_model = powerrec_model(powerhead_model.output)
    # # 组合成新模型
    # model = Model(inputs=powerhead_model.input, outputs=new_rec_model, name=f"concat_model")
    #
    # model.summary()
    # keras_model_path = os.path.join(model_root_folder, "tmp.h5")
    # tf.keras.models.save_model(
    #     model, keras_model_path
    # )
    # convert2lite(keras_model_path, os.path.join(model_root_folder, model_name))
