"""
模型转tflite脚本
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.models import Model


def convert2lite(model, save_path):
    print(save_path)
    # assert os.path.exists(save_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    print("保存完成")


def convert2lite_pb(model, input_shape, save_path):
    print(save_path)

    run_model = tf.function(lambda x: model(x))
    # concrete_func = run_model.get_concrete_function(
    #     tf.TensorSpec([1 * 32 * 229 * 2], model.inputs[0].dtype))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(input_shape, model.inputs[0].dtype))
    # model directory.
    MODEL_DIR = os.path.split(save_path)[0]
    model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    print(f"保存完成, path:{save_path}")


def convert2lite_lstm(model, infer_shape, save_path, reshape=False):
    print(save_path)
    run_model = tf.function(lambda x: model(x))
    # This is important, let's fix the input size.
    BATCH_SIZE = infer_shape[0]  # 1
    STEPS = infer_shape[1]  # 8
    INPUT_SIZE = infer_shape[2]  # 224

    if reshape:
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([BATCH_SIZE * STEPS * INPUT_SIZE], model.inputs[0].dtype))
    else:
        # concrete_func = run_model.get_concrete_function(
        #     tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([STEPS, INPUT_SIZE], model.inputs[0].dtype))
    # model directory.
    MODEL_DIR = os.path.split(save_path)[0]
    model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    print(f"保存完成, path:{save_path}")


if __name__ == '__main__':
    # 测试
    from tensorflow.keras.models import load_model

    # model_root_folder = "/data/projects/BGMcloak/models_files"
    # rec_model_path = os.path.join(model_root_folder, "rec_20220420_3.h5")
    # model = load_model(rec_model_path)

    # model_name = 'rec_pan.tflite'
    # model_root_folder = "/data/projects/BGMcloak/model_files"
    # from models_powerrec.detector import build_PowerRec
    # model = build_PowerRec(input_shape=(24, 224, 1), alpha=1.5, infer_mode=True)
    #
    # convert2lite(model, os.path.join(model_root_folder, model_name))

    # model_name = 'rec_pan_8_f.tflite'
    # model_root_folder = "/data/projects/BGMcloak/model_files"
    # from models_powerrec.detector import build_PowerRec
    # model = build_PowerRec(input_shape=(8, 224, 1), alpha=2.0, infer_mode=True, focus_output=True)

    # model_name = 'fcn.tflite'
    # model_root_folder = "/data/projects/BGMcloak/model_files"
    # from models_powerrec.fcn import FCN
    # model = FCN(input_shape=(8, 224, 1), alpha=1.0)

    # from models_powerrec.powerrec import PowerRec
    #
    # model_name = 'rec_pr.tflite'
    # model_root_folder = "/data/projects/BGMcloak/model_files"
    # model = PowerRec(input_shape=(8, 224, 1), alpha=0.65, channel_num=96)
    # convert2lite(model, os.path.join(model_root_folder, model_name))

    # powerhead_model测试
    # from models.handcraft_model import powerhead_model_4
    #
    # model = powerhead_model_4(img_size=(32, 224), alpha=0.5)
    # model = Model(model.inputs, model.outputs[0])  # 去掉辅助分支
    #
    # model_name = 'powerhead_model_4.tflite'
    # model_root_folder = "/data/projects/BGMcloak/model_files"
    # convert2lite(model, os.path.join(model_root_folder, model_name))
    #
    # from models.handcraft_model import powerhead_model_5
    #
    # model = powerhead_model_5(img_size=(32, 229), alpha=0.3)
    # model = Model(model.inputs, model.outputs[0])  # 去掉辅助分支
    #
    # model_name = 'powerhead_model_5.tflite'
    # model_root_folder = "/data/projects/BGMcloak/model_files"
    # convert2lite(model, os.path.join(model_root_folder, model_name))

    # from models_powerunionrec.powerunionrec import PowerUnionRec, flatten_input_output
    #
    # model_name = 'powerunionrec_20220621_1.tflite'  # 调整为线上输入
    # model_root_folder = "/data/projects/BGMcloak/model_files"
    # # model = PowerUnionRec(input_shape=(32, 229, 2), alpha=1.5)
    # model = load_model("/data1/projects/BGMcloak/train_output_models/powerunionrec_20220615_1_76_0.20298.h5",
    #                    compile=False)
    # # model.summary()
    # # convert2lite(model, os.path.join(model_root_folder, model_name))
    #
    # # 调整
    # model = flatten_input_output(model, (32, 229, 2))  # todo 处理模型输入输出以适应线上
    # model.summary()
    # convert2lite_pb(model, os.path.join(model_root_folder, model_name))
    #

    # from models_powerrec.powerrect_crnn import PowerRec_CRNN
    # # CRNN测试
    # model_name = 'PowerRec_CRNN_20220526_2.tflite'
    # model_root_folder = "/data/projects/BGMcloak/model_files"
    # # 空模型
    # # model = PowerRec_CRNN(input_shape=(8, 224, 1), decoder_dim=256) # 256 6M
    # rec_model_path = os.path.join("/data/projects/BGMcloak/train_output_models/powerrec_20220526_2_46_0.00531.h5")
    # model = load_model(rec_model_path, compile=False)

    # # 输入添加resize
    # from tensorflow.keras.layers import Input
    #
    # inputs = Input(shape=8 * 224)
    # # 输出添加reshape
    # x = tf.reshape(inputs, (1, 8, 224, 1))
    # x = model(x)
    # x = tf.squeeze(x)
    # model = Model(inputs=inputs, outputs=x, name=f"reshape_model")
    #
    # # 训练好的
    # convert2lite_lstm(model, os.path.join(model_root_folder, model_name))

    # from models_crnn.crnn import GOOGLE_CRNN
    #
    # model_name = 'google_crnn.tflite'  # 调整为线上输入
    # model_root_folder = "/data/projects/BGMcloak/model_files"
    #
    # from tensorflow.keras.layers import Input
    #
    #
    #
    # model = GOOGLE_CRNN(input_shape=(8, 229, 1))
    #
    # inputs = Input(shape=8 * 229)
    # # 输出添加reshape
    # x = tf.reshape(inputs, (1, 8, 229, 1))
    # x = model(x)
    # x = tf.squeeze(x)
    # model = Model(inputs=inputs, outputs=x, name=f"reshape_model")
    #
    # model.summary()
    # convert2lite_lstm(model, (1, 8, 229), os.path.join(model_root_folder, model_name), reshape=True)

    from models_forceconcat.force_concat_model import light_mobilenet, midi_spec_head,ForceConcatModel
    from tensorflow import keras

    model_root_folder = "/data/projects/BGMcloak/model_files"
    # model_name = 'light_powerhead.tflite'  # 调整为线上输入
    model_name = 'forceconcat.tflite'  # 调整为线上输入


    # save_path = os.path.join(model_root_folder, model_name)
    # input_shape = (32, 229, 2)
    #
    # inputs = keras.Input(shape=input_shape)
    # mbv3 = light_mobilenet(inputs, 0.5)
    # midi_spec = midi_spec_head(mbv3)
    # model = Model(inputs, midi_spec, name="light_powerhead")
    # model.summary()
    # convert2lite(model, save_path)

    save_path = os.path.join(model_root_folder, model_name)
    input_shape = (32 * 229 * 2)
    inputs = keras.Input(shape=input_shape)
    x = tf.reshape(inputs, (1, 32, 229, 2))
    # train_model_path = "/data1/projects/BGMcloak/225200-8/saved_model/1651392267"
    train_model_path = "/data1/projects/BGMcloak/models_forceconcat/online_rec_model.h5"

    model = ForceConcatModel((32, 229, 2), train_model_path, alpha=0.5, inference_mode=True)

    x = model(x)
    model = Model(inputs=inputs, outputs=x, name=f"reshaped_model")

    model.summary()
    convert2lite_pb(model, input_shape, save_path)