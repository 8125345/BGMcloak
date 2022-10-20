# 注意手工学习率计划与keras默认学习率计划不可同时使用，当前脚本已经关闭手工sheduler了，下面函数都没有用

# 数据集整体数量控制，充分利用被下采样的数据集
mul_times = 1

# 训练时使用的数据集
train_dataset = [
    # ("ai-tagging", 2. * mul_times),  # 18126
    # ("high-note", 1. * mul_times),  # 2388
    # ("maestro", 1. * mul_times),  # 34845
    #
    # # ("Qingchen_bgm_delay_0", 5. * mul_times),  # 1926，无演奏，清晨BGM
    #
    # # # bgm -100 单路模型不需要使用
    # # ("ai-tagging_piano", 0.1 * mul_times),  # 25354
    # # ("high-note_piano", 0.5 * mul_times),  # 3159
    # # ("maestro-v3.0.0_piano", 0.1 * mul_times),  # 37543
    # # ("Qingchen_piano", 0.5 * mul_times),  # 1767
    # # ("Peilian_piano", 0.5 * mul_times),  # 3066
    #
    # # 不同音色库合成
    # ("Qingchen_xml", 2. * mul_times),  # 1767
    # ("Peilian_xml", 2. * mul_times),  # 3066
    # ("Qingchen_xml_SC55", 2. * mul_times),  # 1755
    # ("Peilian_xml_SC55", 2. * mul_times),  # 3113
    #
    # # ！！！！！！！！上述数据集BGM被污染！！！！！！！！！！！！！！！！！！！

    # 清晨实录
    # 共计5548
    ("bgm_record_20220721-22", 4. * mul_times),  # 1299
    ("bgm_record_20220725", 4. * mul_times),  # 913
    ("bgm_record_20220726", 4. * mul_times),  # 878
    ("bgm_record_20220727", 4. * mul_times),  # 1358
    ("bgm_record_20220728", 4. * mul_times),  # 1100

    # 新加
    # 4到5月共计18583
    ("bgm_record_20220411_train", 2. * mul_times),  # 3487
    ("bgm_record_20220414_train", 2. * mul_times),  # 2586
    ("bgm_record_20220415_train", 2. * mul_times),  # 2259
    ("bgm_record_20220420_train", 2. * mul_times),  # 843
    ("bgm_record_20220428_train", 2. * mul_times),  # 3850
    ("bgm_record_20220429_train", 2. * mul_times),  # 4095
    ("bgm_record_20220516_train", 2. * mul_times),  # 796
    ("bgm_record_20220517_train", 2. * mul_times),  # 667

    # 7302
    ("bgm_record_peilian_train", 2. * mul_times),  # 1194
    ("bgm_specail_train", 2. * mul_times),  # 6108

    # 混instapiano
    # 9349
    ("high-note_-20_-5_noised_train", 2. * mul_times),  # 3768
    ("single-note_-20_-5_noised_train", 2. * mul_times),  # 5581

    # 8月共计4623
    ("bgm_record_20220822_train", 2. * mul_times),  # 515
    ("bgm_record_20220823_train", 2. * mul_times),  # 1287
    ("bgm_record_20220824_train", 2. * mul_times),  # 1404
    ("bgm_record_20220825_train", 2. * mul_times),  # 1194
    ("bgm_record_20220826_train", 2. * mul_times),  # 223

    # 纯负样本
    # 21481
    ("negbgm_train", 4. * mul_times),  # 11923 录制BGM
    ("negbgm_train_ori", 4. * mul_times),  # 2961，原声BGM
    ("noise_wo_bgm", 2. * mul_times),  # 6597，环境噪声

    # 修正后的数据集
    ("ai-tagging_new", 3. * mul_times),  # 5868
    ("high-note_new", 3. * mul_times),  # 3920
    ("low-note_new", 10. * mul_times),  # 153
    ("single-note_new", 3. * mul_times),  # 5581

    ("maestro_new", 1. * mul_times),  # 70419

]
# cp forceconcat_20220902_0_default.tflite /data/projects/test_models

# 基础配置，如果train_config中存在同名变量，会用train_config变量覆盖basic_config配置
basic_config = {
    "model_root": "/data/projects/BGMcloak/train_output_models",
    "train_log_root": "/data/projects/BGMcloak/train_log",
    "train_comment_root": "/data1/projects/BGMcloak/train_comment",
}

# ==================================================================================
# ==================================================================================

# 本次训练配置

# 20220921
train_config = {
    "model_name": f"forceconcat_20220921_2",
    "gpuids": [3],
    "train_batchsize": (512 + 512) * 1,
    "val_batchsize": (512 + 512) * 1,
    "add_maestro": True,
    "pos_weight": 50,
    "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220916_2.h5",
    "model_structure": "singlepro-8",  # forceconcat_de
    "lr": 0.001 / 10,  # 初始学习率
    "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220916_0_train",
    "rec_loss_fun": "weighted_bce",
    "comment": "单路，input8，增加实录数据，修正BGM",
}

# train_config = {
#     "model_name": f"forceconcat_20220921_1",
#     "gpuids": [2],
#     "train_batchsize": (512 + 512) * 1,
#     "val_batchsize": (512 + 512) * 1,
#     "add_maestro": True,
#     "pos_weight": 150,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220916_2.h5",
#     "model_structure": "singlepro-8",  # forceconcat_de
#     "lr": 0.001 / 10,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220916_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "单路，input8，增加实录数据，修正BGM",
# }

# train_config = {
#     "model_name": f"forceconcat_20220921_0",
#     "gpuids": [0],
#     "train_batchsize": (512 + 512) * 1,
#     "val_batchsize": (512 + 512) * 1,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220916_2.h5",
#     "model_structure": "singlepro-8",  # forceconcat_de
#     "lr": 0.001 / 10,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220916_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "单路，input8，增加实录数据，修正BGM",
# }

# ==================================================================================
# 20220916
# train_config = {
#     "model_name": f"forceconcat_20220916_2",
#     "gpuids": [3],
#     "train_batchsize": (512 + 512) * 1,
#     "val_batchsize": (512 + 512) * 1,
#     "add_maestro": True,
#     "pos_weight": 50,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220908_3.h5",
#     "model_structure": "singlepro-8",  # forceconcat_de
#     "lr": 0.001 / 10,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220916_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "单路，input8，增加实录数据",
# }

# train_config = {
#     "model_name": f"forceconcat_20220916_1",
#     "gpuids": [2],
#     "train_batchsize": (512 + 512) * 1,
#     "val_batchsize": (512 + 512) * 1,
#     "add_maestro": True,
#     "pos_weight": 150,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220908_3.h5",
#     "model_structure": "singlepro-8",  # forceconcat_de
#     "lr": 0.001 / 10,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220916_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "单路，input8，增加实录数据",
# }
# train_config = {
#     "model_name": f"forceconcat_20220916_0",
#     "gpuids": [0],
#     "train_batchsize": (512 + 512) * 1,
#     "val_batchsize": (512 + 512) * 1,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220908_3.h5",
#     "model_structure": "singlepro-8",  # forceconcat_de
#     "lr": 0.001 / 10,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220916_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "单路，input8，增加实录数据",
# }
# ==================================================================================
# 20220908
# train_config = {
#     "model_name": f"forceconcat_20220908_3",
#     "gpuids": [3],
#     "train_batchsize": (512 + 512) * 1,
#     "val_batchsize": 512 * 1,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220906_1.h5",
#     # singlepro-8 singlepro forceconcat_de_base
#     "model_structure": "singlepro-8",  # forceconcat_de
#     "lr": 0.001 / 10,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "单路，input8，大权重",
# }

# train_config = {
#     "model_name": f"forceconcat_20220908_2",
#     "gpuids": [2],
#     "train_batchsize": (512 + 512) * 1,
#     "val_batchsize": 512 * 1,
#     "add_maestro": True,
#     "pos_weight": 150,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220906_1.h5",
#     # singlepro-8 singlepro forceconcat_de_base
#     "model_structure": "singlepro-8",  # forceconcat_de
#     "lr": 0.001 / 10,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "单路，input8，大权重",
# }


# train_config = {
#     "model_name": f"forceconcat_20220908_1",
#     "gpuids": [1],
#     "train_batchsize": (512 + 256) * 1,
#     "val_batchsize": 512 * 1,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220906_1.h5",
#     # singlepro-8 singlepro forceconcat_de_base
#     "model_structure": "singlepro-8",  # forceconcat_de
#     "lr": 0.001 / 10,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "单路，input8，大权重",
# }

# train_config = {
#     "model_name": f"forceconcat_20220908_0",
#     "gpuids": [0],
#     "train_batchsize": (512) * 2,
#     "val_batchsize": 512 * 1,
#     "add_maestro": True,
#     "pos_weight": 150,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220906_1.h5",
#     # singlepro-8 singlepro forceconcat_de_base
#     "model_structure": "singlepro-8",  # forceconcat_de
#     "lr": 0.001 / 10,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "单路，input8，大权重",
# }
# ==================================================================================
happy_split = 666
# # 20220906
# train_config = {
#     "model_name": f"forceconcat_20220906_1",
#     "gpuids": [1, 3],
#     "train_batchsize": 512 *3,
#     "val_batchsize": 512 * 1,
#     "add_maestro": True,
#     "pos_weight": 50,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/singlepro-8.h5",
#     # singlepro-8 singlepro forceconcat_de_base
#     "model_structure": "singlepro-8",  # forceconcat_de
#     "lr": 0.001 / 2,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "单路，去除-100，input8",
# }

# train_config = {
#     "model_name": f"forceconcat_20220906_0",
#     "gpuids": [0, 2],
#     "train_batchsize": 512 * 4,
#     "val_batchsize": 512 * 1,
#     "add_maestro": True,
#     "pos_weight": 50,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220906_0.h5",  # singlepro forceconcat_de_base
#     "model_structure": "singlepro",  # forceconcat_de
#     "lr": 0.001/2,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "单路，去除-100",
# }


# 20220905
# 单路模型
# train_config = {
#     "model_name": f"forceconcat_20220905_2",
#     "gpuids": [3],
#     "train_batchsize": 512 * 2,
#     "val_batchsize": 512 * 1,
#     "add_maestro": True,
#     "pos_weight": 50,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220905_2.h5",  # singlepro forceconcat_de_base
#     "model_structure": "singlepro",  # forceconcat_de
#     "lr": 0.001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "数据集*2，模型去掉bottleneck, bise，单路",
# }

# train_config = {
#     "model_name": f"forceconcat_20220905_1",
#     "gpuids": [1],
#     "train_batchsize": 512 * 1,
#     "val_batchsize": 512 * 1,
#     "add_maestro": True,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220905_1.h5",  # singlepro forceconcat_de_base
#     "model_structure": "singlepro",  # forceconcat_de
#     "lr": 0.001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "数据集*2，模型去掉bottleneck, bise，单路",
# }

# # # 双路模型
# train_config = {
#     "model_name": f"forceconcat_20220905_0",
#     "gpuids": [0, 2],
#     "train_batchsize": 512 * 3,
#     "val_batchsize": 512 * 2,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220905_0.h5",  # forceconcat_de_base
#     "model_structure": "doublepro",  # forceconcat_de
#     "lr": 0.001 / 3,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "数据集*2，模型去掉bottleneck, bise",
# }


#
# # 20220902
#
# train_config = {
#     "model_name": f"forceconcat_20220902_5",
#     "gpuids": [2, 3],
#     "train_batchsize": 512 * 2,
#     "val_batchsize": 512 * 2,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/doublepro.h5",  # forceconcat_de_base
#     "model_structure": "doublepro",  # forceconcat_de
#     "lr": 0.001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "数据集*2，模型去掉bottleneck",
# }

# train_config = {
#     "model_name": f"forceconcat_20220902_4",
#     "gpuids": [0],
#     "train_batchsize": 512 * 1,
#     "val_batchsize": 512 * 1,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220902_4.h5",  # forceconcat_de_base
#     "model_structure": "doublepro",  # forceconcat_de
#     "lr": 0.0001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "数据集*2，模型去掉bottleneck",
# }


# train_config = {
#     "model_name": f"forceconcat_20220902_3",
#     "gpuids": [1],
#     "train_batchsize": 512 * 1,
#     "val_batchsize": 512 * 1,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220902_2.h5",  # forceconcat_de_base
#     "model_structure": "doublepro",  # forceconcat_de
#     "lr": 0.0001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "数据集*2，模型去掉bottleneck",
# }

# train_config = {
#     "model_name": f"forceconcat_20220902_2",
#     "gpuids": [0, 1, 2, 3],
#     "train_batchsize": 512 * 4,
#     "val_batchsize": 512 * 4,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220902_1.h5",  # forceconcat_de_base
#     "model_structure": "doublepro",  # forceconcat_de
#     "lr": 0.001 / 4,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "数据集*2，模型去掉bottleneck",
# }

# train_config = {
#     "model_name": f"forceconcat_20220902_0",
#     "gpuids": [0, 2],
#     "train_batchsize": 512*4,
#     "val_batchsize": 512*2,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220829_2.h5",  # forceconcat_de_base
#     "model_structure": "forceconcat",  # forceconcat_de
#     "lr": 0.001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "数据集*2",
# }


# 20220901
# train_config = {
#     "model_name": f"forceconcat_20220901_2",
#     "gpuids": [1],
#     "train_batchsize": (512) * 1,
#     "val_batchsize": (512) * 1,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220901_1.h5",  # doublepro_large forceconcat_de_base # todo
#     "model_structure": "doublepro",  # forceconcat_de
#     "lr": 0.0001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "更改模型结构",
# }
# train_config = {
#     "model_name": f"forceconcat_20220901_1",
#     "gpuids": [0, 1, 2, 3],
#     "train_batchsize": (512 + 256) * 4,
#     "val_batchsize": (512 + 256) * 1,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220901_0.h5",  # doublepro_large forceconcat_de_base # todo
#     "model_structure": "doublepro",  # forceconcat_de
#     "lr": 0.001 / 3,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "更改模型结构",
# }
# # 20220831
# train_config = {
#     "model_name": f"forceconcat_20220831_0",
#     "gpuids": [0, 2, 3],
#     "train_batchsize": (512 + 256) * 3,
#     "val_batchsize": (512 + 256) * 1,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/doublepro_large.h5",  # doublepro_large forceconcat_de_base # todo
#     "model_structure": "doublepro",  # forceconcat_de
#     "lr": 0.001 * 3,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "更改模型结构",
# }

# 20220830
# train_config = {
#     "model_name": f"forceconcat_20220830_0",
#     "gpuids": [0, 2, 3],
#     "train_batchsize": (512 + 256) * 3,
#     "val_batchsize": (512 + 256) * 1,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220830_0.h5",  # forceconcat_de_base # todo
#     "model_structure": "doublepro",  # forceconcat_de
#     "lr": 0.001 * 3,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "更改模型结构",
# }

# 20220829
# train_config = {
#     "model_name": f"forceconcat_20220829_3",
#     "gpuids": [3],
#     "train_batchsize": 512 * 3,
#     "val_batchsize": 512 * 3,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220826_0.h5",  # forceconcat_de_base
#     "model_structure": "forceconcat",  # forceconcat_de
#     "lr": 0.001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "去除人工演奏数据",
# }

# train_config = {
#     "model_name": f"forceconcat_20220829_2",
#     "gpuids": [2],
#     "train_batchsize": 512*2,
#     "val_batchsize": 512*2,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220826_0.h5",  # forceconcat_de_base
#     "model_structure": "forceconcat",  # forceconcat_de
#     "lr": 0.001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "大权重",
# }


# train_config = {
#     "model_name": f"forceconcat_20220829_1",
#     "gpuids": [1],
#     "train_batchsize": 512*2,
#     "val_batchsize": 512*2,
#     "add_maestro": True,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220826_0.h5",  # forceconcat_de_base
#     "model_structure": "forceconcat",  # forceconcat_de
#     "lr": 0.001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "大权重",
# }


# train_config = {
#     "model_name": f"forceconcat_20220829_0",
#     "gpuids": [0],
#     "train_batchsize": 512*3,
#     "val_batchsize": 512*3,
#     "add_maestro": True,
#     "pos_weight": 650,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220826_0.h5",  # forceconcat_de_base
#     "model_structure": "forceconcat",  # forceconcat_de
#     "lr": 0.001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "大权重",
# }

# 20220826
# train_config = {
#     "model_name": f"forceconcat_20220826_3",
#     "gpuids": [3],
#     "train_batchsize": 512*3,
#     "val_batchsize": 512*3,
#     "add_maestro": True,
#     "pos_weight": 50,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220826_3.h5",  # forceconcat_de_base
#     "model_structure": "forceconcat_de",  # forceconcat_de
#     "lr": 0.001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "多任务学习，带有决策头",
# }


# train_config = {
#     "model_name": f"forceconcat_20220826_2",
#     "gpuids": [2],
#     "train_batchsize": 512*3,
#     "val_batchsize": 512*3,
#     "add_maestro": True,
#     "pos_weight": 10,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220819_8.h5",
#     "model_structure": "forceconcat",  # forceconcat_de
#     "lr": 0.001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "调整数据集比例，提升和弦识别率，降低正样本比例",
# }


# train_config = {
#     "model_name": f"forceconcat_20220826_1",
#     "gpuids": [1],
#     "train_batchsize": 512*2,
#     "val_batchsize": 512*2,
#     "add_maestro": True,
#     "pos_weight": 50,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220819_8.h5",
#     "model_structure": "forceconcat",  # forceconcat_de
#     "lr": 0.001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "调整数据集比例，提升和弦识别率，降低正样本比例",
# }


# train_config = {
#     "model_name": f"forceconcat_20220826_0",
#     "gpuids": [0],
#     "train_batchsize": 512*3,
#     "val_batchsize": 512*3,
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220819_8.h5",
#     "model_structure": "forceconcat",  # forceconcat_de
#     "lr": 0.001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "调整数据集比例，提升和弦识别率",
# }

# ==================================================================================
# ==================================================================================
# 20220819
# train_config = {
#     "model_name": f"forceconcat_20220819_16",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 650,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220819_8.h5",
#     "model_structure": "forceconcat_de",
#     "lr": 0.001,  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#
#     "comment": "测试",
# }

# ==================================================================================
# ==================================================================================
# 20220819
# train_config = {
#     "model_name": f"forceconcat_20220819_16",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 650,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220819_8.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "增加召回测试",
# }

# train_config = {
#     "model_name": f"forceconcat_20220819_15",
#     "gpuids": [2],
#     "add_maestro": False,
#     "pos_weight": 600,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220819_8.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "增加召回测试",
# }


# train_config = {
#     "model_name": f"forceconcat_20220819_14",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 550,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220819_8.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "增加召回测试",
# }

# train_config = {
#     "model_name": f"forceconcat_20220819_13",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220819_8.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "噪声置换测试",
# }

# ==================================================================================
# ==================================================================================
# 20220819

# train_config = {
#     "model_name": f"forceconcat_20220819_12",
#     "gpuids": [2],
#     "add_maestro": False,
#     "pos_weight": 650,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220819_8.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "增加召回测试",
# }

# train_config = {
#     "model_name": f"forceconcat_20220819_11",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 550,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220819_8.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "增加召回测试",
# }
# train_config = {
#     "model_name": f"forceconcat_20220819_10",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 600,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220819_8.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "增加召回测试",
# }


# train_config = {
#     "model_name": f"forceconcat_20220819_9",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "去除无BGM所有演奏（包含环境噪声），+清晨合成音色",
# }

# train_config = {
#     "model_name": f"forceconcat_20220819_8",
#     "gpuids": [2],
#     "add_maestro": False,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "去除无BGM所有演奏（包含环境噪声），+清晨合成音色",
# }

# train_config = {
#     "model_name": f"forceconcat_20220819_7",
#     "gpuids": [1],
#     "add_maestro": True,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "去除无BGM所有演奏（包含环境噪声），建立新基线",
# }


# train_config = {
#     "model_name": f"forceconcat_20220819_6",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "去除无BGM所有演奏（包含环境噪声），建立新基线",
# }

# ==================================================================================
# ==================================================================================
# train_config = {
#     "model_name": f"forceconcat_20220819_5",
#     "gpuids": [2],
#     "add_maestro": True,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "去除无BGM演奏，增加电子合成音，提高音色兼容",
# }

# train_config = {
#     "model_name": f"forceconcat_20220819_4",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "去除无BGM演奏，增加电子合成音，提高音色兼容",
# }


# train_config = {
#     "model_name": f"forceconcat_20220819_3",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 700,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "去除无BGM演奏",
# }


# train_config = {
#     "model_name": f"forceconcat_20220819_2",
#     "gpuids": [2],
#     "add_maestro": True,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "去除无BGM演奏",
# }


# train_config = {
#     "model_name": f"forceconcat_20220819_1",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 700,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "去除无BGM演奏",
# }

# train_config = {
#     "model_name": f"forceconcat_20220819_0",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220819_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "去除无BGM演奏",
# }


# ==================================================================================
# ==================================================================================
# # 20220817
#
# train_config = {
#     "model_name": f"forceconcat_20220817_0",
#     "gpuids": [0],
#     "add_maestro": True,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220817_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "扩大无BGM数据比例",
# }

# ==================================================================================
# ==================================================================================
# 20220812

# train_config = {
#     "model_name": f"forceconcat_20220812_3",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 300,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220812_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "加清晨电子键盘音色",
# }


# train_config = {
#     "model_name": f"forceconcat_20220812_2",
#     "gpuids": [2],
#     "add_maestro": False,
#     "pos_weight": 300,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220812_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "加清晨电子键盘音色，减小pw",
# }

# train_config = {
#     "model_name": f"forceconcat_20220812_1",
#     "gpuids": [1],
#     "add_maestro": True,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220812_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "加清晨电子键盘音色",
# }

# train_config = {
#     "model_name": f"forceconcat_20220812_0",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220811_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220812_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "加清晨电子键盘音色",
# }
# ==================================================================================
# ==================================================================================
# 20220811
# train_config = {
#     "model_name": f"forceconcat_20220811_5",
#     "gpuids": [1],
#     "add_maestro": True,
#     "pos_weight": 700,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220810_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220811_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "加maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220811_4",
#     "gpuids": [0],
#     "add_maestro": True,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220810_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220811_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "加maestro",
# }


# train_config = {
#     "model_name": f"forceconcat_20220811_3",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 400,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220810_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220811_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "加maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220811_2",
#     "gpuids": [2],
#     "add_maestro": True,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220810_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220811_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "加maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220811_1",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 700,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220810_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220722_1_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "对forceconcat_20220810_1中的pw权重进行扰动",
# }

# train_config = {
#     "model_name": f"forceconcat_20220811_0",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 400,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220810_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220722_1_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "对forceconcat_20220810_1中的pw权重进行扰动",
# }

# ==================================================================================
# ==================================================================================
# 20220810

# train_config = {
#     "model_name": f"forceconcat_20220810_1",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220810_0.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_bce",
#     "comment": "添加新数据，抑制无音乐播放时的识别错误，调大正样本权重避免低召回",
# }

# train_config = {
#     "model_name": f"forceconcat_20220810_0",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220810_0.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_bce",
#     "comment": "添加新数据，抑制无音乐播放时的识别错误",
# }


# ==================================================================================
# ==================================================================================
# 20220808

# train_config = {
#     "model_name": f"forceconcat_20220808_3",
#     "gpuids": [3],
#     "add_maestro": False,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220804_0.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "center_weighted_bce",
#     "comment": "实验组，forceconcat_20220808_2，loss边缘降权",
# }


# train_config = {
#     "model_name": f"forceconcat_20220808_2",
#     "gpuids": [2],
#     "add_maestro": False,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220804_0.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_bce",
#     "comment": "对照组，普通wbce",
# }


# train_config = {
#     "model_name": f"forceconcat_20220808_1",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220804_0.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "center_weighted_bce",
#     "comment": "实验组，对比forceconcat_20220808_0，loss边缘降权",
# }

# train_config = {
#     "model_name": f"forceconcat_20220808_0",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220804_0.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_bce",
#     "comment": "对照组，普通bce",
# }

# ==================================================================================
# ==================================================================================
# 20220805

# train_config = {
#     "model_name": f"forceconcat_20220805_5",
#     "gpuids": [2],
#     "add_maestro": False,
#     "pos_weight": 80,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220804_0.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_fl",
#     "comment": "cosin lr 0.001，pos weight 80, weighted_fl, wo add_maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220805_4",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 160,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220804_0.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_fl",
#     "comment": "cosin lr 0.001，pos weight 160, weighted_fl, wo add_maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220805_3",
#     "gpuids": [3],
#     "add_maestro": False,
#     "pos_weight": 30,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220804_0.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_fl",
#     "comment": "cosin lr 0.001，pos weight 30, weighted_fl, wo add_maestro",
# }

# ---------------------------------------------------------------------------------
# train_config = {
#     "model_name": f"forceconcat_20220805_2",
#     "gpuids": [2],
#     "add_maestro": False,
#     "pos_weight": 400,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220805_2.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_bce",
#     "comment": "cosin lr 0.001，pos weight 400, weighted_bce, wo add_maestro",
# }


# train_config = {
#     "model_name": f"forceconcat_20220805_1",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 700,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220805_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_bce",
#     "comment": "cosin lr 0.001，pos weight 700, weighted_bce, wo add_maestro",
# }


# train_config = {
#     "model_name": f"forceconcat_20220805_0",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 500,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220805_0.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_bce",
#     "comment": "cosin lr 0.001，pos weight 500, weighted_bce, wo add_maestro",
# }

# ==================================================================================
# ==================================================================================
# 20220804


# train_config = {
#     "model_name": f"forceconcat_20220804_5",
#     "gpuids": [2],
#     "add_maestro": False,
#     "pos_weight": 5,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220730_6.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_bce",
#     "comment": "cosin lr 0.001，pos weight 5, weighted_bce, wo add_maestro",
# }


# train_config = {
#     "model_name": f"forceconcat_20220804_4",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 10,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220730_6.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_bce",
#     "comment": "cosin lr 0.001，pos weight 10, weighted_bce, wo add_maestro",
# }


# train_config = {
#     "model_name": f"forceconcat_20220804_3",
#     "gpuids": [3],
#     "add_maestro": False,
#     "pos_weight": 40,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220730_6.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_bce",
#     "comment": "cosin lr 0.001，pos weight 40, weighted_bce, wo add_maestro",
# }


# train_config = {
#     "model_name": f"forceconcat_20220804_2",
#     "gpuids": [2],
#     "add_maestro": False,
#     "pos_weight": 80,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220730_6.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_bce",
#     "comment": "cosin lr 0.001，pos weight 80, weighted_bce, wo add_maestro",
# }


# train_config = {
#     "model_name": f"forceconcat_20220804_1",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 150,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220804_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_bce",
#     "comment": "cosin lr 0.001，pos weight 150, weighted_bce, wo add_maestro",
# }


# train_config = {
#     "model_name": f"forceconcat_20220804_0",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220730_6.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_bce",
#     "comment": "cosin lr 0.001，pos weight 250, weighted_bce, wo add_maestro",
# }

# ==================================================================================
# ==================================================================================
# 20220731
# train_config = {
#     "model_name": f"forceconcat_20220731_3",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 30,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220731_3.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_fl",
#     "comment": "lr 0.001，pos weight 30, fl, add_maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220731_2",
#     "gpuids": [2],
#     "add_maestro": False,
#     "pos_weight": 160,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220731_2.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_fl",
#     "comment": "lr 0.001，pos weight 160, fl",
# }

# train_config = {
#     "model_name": f"forceconcat_20220731_1",
#     "gpuids": [1],
#     "add_maestro": True,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220730_6.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_fl",
#     "comment": "lr 0.001，pos weight 1, fl, add_maestro",
# }


# train_config = {
#     "model_name": f"forceconcat_20220731_0",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220730_6.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "rec_loss_fun": "weighted_fl",
#     "comment": "lr 0.001，pos weight 1, fl",
# }

# ==================================================================================
# ==================================================================================

# 20220730

# train_config = {
#     "model_name": f"forceconcat_20220730_6",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 250,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220730_6.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 250:1, delay, +xml",
# }


# train_config = {
#     "model_name": f"forceconcat_20220730_5",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 80,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220730_3.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 80:1, delay, add_maestro, +xml",
# }

# train_config = {
#     "model_name": f"forceconcat_20220730_4",
#     "gpuids": [2],
#     "add_maestro": False,
#     "pos_weight": 80,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220730_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 80:1, delay, +xml",
# }


# train_config = {
#     "model_name": f"forceconcat_20220730_3",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 30,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220729_3.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 30:1, delay,add_maestro, +xml",
# }

# train_config = {
#     "model_name": f"forceconcat_20220730_2",
#     "gpuids": [2],
#     "add_maestro": True,
#     "pos_weight": 10,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220729_2.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 10:1, delay, add_maestro, +xml",
# }

# train_config = {
#     "model_name": f"forceconcat_20220730_1",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 30,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220729_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 30:1, delay, +xml",
# }

# train_config = {
#     "model_name": f"forceconcat_20220730_0",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 10,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220729_0.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 10:1, delay, +xml",
# }

# ==================================================================================
# ==================================================================================


# 202220729

# train_config = {
#     "model_name": f"forceconcat_20220729_3",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 30,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220729_3.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 30:1, delay, 滚动加数据,add_maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220729_2",
#     "gpuids": [2],
#     "add_maestro": True,
#     "pos_weight": 10,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220729_2.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 10:1, delay, 滚动加数据,add_maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220729_1",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 30,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220729_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 30:1, delay, 滚动加数据",
# }

# train_config = {
#     "model_name": f"forceconcat_20220729_0",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 10,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220729_0.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 10:1, delay, 滚动加数据",
# }


# ==================================================================================
# ==================================================================================


# 20220728

# train_config = {
#     "model_name": f"forceconcat_20220728_11",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 10,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220728_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 10:1, add_maestro，delay",
# }


# train_config = {
#     "model_name": f"forceconcat_20220728_10",
#     "gpuids": [2],
#     "add_maestro": False,
#     "pos_weight": 10,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220728_4.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 10:1，delay",
# }


# train_config = {
#     "model_name": f"forceconcat_20220728_9",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 5,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220728_3.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 5:1，添加maestro，delay",
# }

# train_config = {
#     "model_name": f"forceconcat_20220728_8",
#     "gpuids": [2],
#     "add_maestro": True,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220728_2.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 1:1，添加maestro，delay",
# }

# train_config = {
#     "model_name": f"forceconcat_20220728_7",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 5,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220728_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 5:1，delay",
# }

# train_config = {
#     "model_name": f"forceconcat_20220728_6",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220728_0.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 1:1，delay",
# }


# =========================
# train_config = {
#     "model_name": f"forceconcat_20220728_5",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 10,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220728_3.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 10:1, add_maestro",
# }


# train_config = {
#     "model_name": f"forceconcat_20220728_4",
#     "gpuids": [2],
#     "add_maestro": False,
#     "pos_weight": 10,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220728_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 10:1",
# }


# train_config = {
#     "model_name": f"forceconcat_20220728_3",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 5,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220727_9.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 5:1，添加maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220728_2",
#     "gpuids": [2],
#     "add_maestro": True,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220727_8.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 1:1，添加maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220728_1",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 5,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220727_7.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 5:1",
# }

# train_config = {
#     "model_name": f"forceconcat_20220728_0",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220727_6.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 1:1",
# }


# # 20220727
# train_config = {
#     "model_name": f"forceconcat_20220727_9",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 5,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220727_5.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 5:1，添加maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220727_8",
#     "gpuids": [2],
#     "add_maestro": True,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220727_4.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 1:1，添加maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220727_7",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 5,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220727_3.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 5:1",
# }

# train_config = {
#     "model_name": f"forceconcat_20220727_6",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220727_2.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 1:1",
# }


# train_config = {
#     "model_name": f"forceconcat_20220727_5",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 5,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220727_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 5:1，添加maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220727_4",
#     "gpuids": [2],
#     "add_maestro": True,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220727_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 1:1，添加maestro",
# }

# train_config = {
#     "model_name": f"forceconcat_20220727_3",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 5,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220727_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "提高学习率到0.001，pos weight 5:1",
# }

# train_config = {
#     "model_name": f"forceconcat_20220727_2",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220727_1.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": no_scheduler,
#     "comment": "测试大学习率学习，提高大学习率到0.001，保持大学习率，观察可以优化到的极限效果",
# }
#

# train_config = {
#     "model_name": f"forceconcat_20220727_1",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220726_3.h5",
#     "lr": 0.001,  # 初始学习率
#     "scheduler": large_lr_scheduler,
#     "comment": "测试大学习率学习，提高大学习率到0.001",
# }
# 239/239 [==============================] - 27s 113ms/step - loss: 0.0035 - auc: 0.9748 - precision: 0.7934 - recall: 0.5744 - recall_1: 0.8164 - val_loss: 0.0037 - val_auc: 0.9751 - val_precision: 0.7984 - val_recall: 0.5476 - val_recall_1: 0.8096 - lr: 3.3333e-06
# train_config = {
#     "model_name": f"forceconcat_20220727_1",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220726_3.h5",
#     "lr": 0.0001,  # 初始学习率
#     "scheduler": scheduler,
#     "comment": "测试大学习率学习，观察loss是否能快速下降",
# }

# 20220726
# train_config = {
#     "model_name": f"forceconcat_20220726_4",
#     "gpuids": [0],
#     "add_maestro": False,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220726_3.h5",
# }

# train_config = {
#     "model_name": f"forceconcat_20220726_5",
#     "gpuids": [1],
#     "add_maestro": False,
#     "pos_weight": 5,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220726_3.h5",
#
# }

# train_config = {
#     "model_name": f"forceconcat_20220726_6",
#     "gpuids": [2],
#     "add_maestro": True,
#     "pos_weight": 1,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220726_3.h5",
#
# }
#
# train_config = {
#     "model_name": f"forceconcat_20220726_7",
#     "gpuids": [3],
#     "add_maestro": True,
#     "pos_weight": 5,
#     "pretrain_model": "/data1/projects/BGMcloak/train_output_models/forceconcat_20220726_3.h5",
#
# }


# model_name = f"forceconcat_20220630_1"# 最优结果
# model_name = f"forceconcat_20220708_1"  # 使用智能陪练模型
# model_name = f"tmp_forceconcat_20220723_1"  # 用于调试的模型
# model_name = f"forceconcat_20220724_1"  # 新数据集 499
# model_name = f"forceconcat_20220725_1"  # 新数据集 499，去除maestro
# model_name = f"tmp_forceconcat_20220725_1"  # 测试504
# model_name = f"forceconcat_20220725_2"  # 504
# model_name = f"forceconcat_20220725_3"  # 504 加权1：10
# model_name = f"forceconcat_20220725_4"  # 504 加权1：100
# model_name = f"forceconcat_20220725_5"  # 499 加权1：10
# model_name = f"tmp_forceconcat_20220726_1"  # forceconcat_20220725_3，解除全部冻结层
# model_name = f"forceconcat_20220726_1"  # forceconcat_20220725_3，解除全部冻结层，504 不加权
# model_name = f"forceconcat_20220726_2"  # forceconcat_20220725_3，解除全部冻结层，504 loss 5
# model_name = f"forceconcat_20220726_3"  # forceconcat_20220726_1，解除全部冻结层，504 loss 10

# pretrain_model = "/data1/projects/BGMcloak/train_output_models/forceconcat_20220725_3.h5"
# pretrain_model = "/data1/projects/BGMcloak/train_output_models/forceconcat_20220726_1-Copy1.h5"  # 无冻结层


# 无预训练
# pretrain_model = None
# lr = (0.0001) / len(gpuids)
# lr = (0.0001) / 3  # todo 下面学习率计划已经写死了，这项暂时不生效
# def scheduler(epoch, lr):
#     # 注意，这里的lr指的是上一轮的lr，不是最初lr
#     if epoch < 20:
#         return 0.01 / 3
#     elif epoch < 50:
#         return 0.001 / 3
#     elif epoch < 100:
#         return 0.0001 / 3
#     else:
#         return 0.00001 / 3
# model_kwarg = {
#     "lr": lr,
#     "compile_pretrain": False,
#     "reconstruct": None,
#     "base_model": "499",
#     # "base_model": "504",
# }

# es_patience = 1000  # early stop
# rlr_patience = 5  # reduce lr  # todo
# rlr_factor = 0.1  # reduce lr 比例
