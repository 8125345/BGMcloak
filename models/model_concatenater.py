"""
模型拼接
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from models.handcraft_model import rec_model, powerhead_model_1
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model


def concat_model(head_model_path, rec_model_path):
    # 背景音模型
    powerhead_model = load_model(head_model_path)
    powerhead_model.summary()

    # 声音识别模型
    rec_model = load_model(rec_model_path)
    rec_model.summary()

    # 把背景音模型的输出传到声音识别的输入
    new_rec_model = rec_model(powerhead_model.output)

    # 组合成新模型
    concat_model = Model(inputs=powerhead_model.input, outputs=new_rec_model, name=f"concat_model")
    return concat_model


if __name__ == '__main__':
    model_root_folder = "/data/projects/BGMcloak/models_files"
    powerhead_model_path = os.path.join(model_root_folder, "handcraft_20220407_1.h5")
    rec_model_path = os.path.join(model_root_folder, "rec_20220407_1.h5")

    model = concat_model(powerhead_model_path, rec_model_path)
    model.summary()