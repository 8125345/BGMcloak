"""
把mel频谱图转换为音频格式，便于校验数据
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import os.path
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import glob
import json
import numpy as np
import soundfile as sf

import tensorflow as tf
from pkl_to_midi import pkl_to_mid

SR = samplerate = 16000


def spec2mel(data):
    spec = librosa.db_to_power(data)
    mel = spec.T

    return mel


def mel2audio(fn, sr=SR, hop_length=512, fmin=30.0, htk=True):
    mel = fn
    audio = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        # n_mels=n_mels,  # n_mels的特征librosa后续模块会从shape反向导出，无需填写
        htk=htk)
    return audio


def parse_np_data(path):
    """
    解析numpy格式文件
    :param path:
    :return:
    """
    data = np.load(path)
    chunk_spec = data[:, : 229]
    bgm_chunk_spec = data[:, 229: 2 * 229]
    onsets_label = data[:, 2 * 229:]
    return bgm_chunk_spec, chunk_spec, onsets_label


def parse_serialized_data(path):
    serialized = tf.io.read_file(path)
    data = tf.io.parse_tensor(serialized, tf.float32)

    feature = data[:, : 293120]
    midi = data[:, 293120:]

    feature = tf.reshape(feature, (640, 229, 2))
    midi = tf.reshape(midi, (640, 88))

    # concat = tf.concat((feature[:, :, 0], feature[:, :, 1], midi), axis=1)  # 640 * (229 + 229 + 88)
    bgm_chunk_spec = feature[:, :, 1]
    chunk_spec = feature[:, :, 1]
    return bgm_chunk_spec, chunk_spec, midi


def recover_seg(path, dst_folder_path, dst_name, mode, save_midi=True):
    """

    :param dst_name:
    :param path:
    :param dst_folder_path:
    :param mode:
    :return:
    """
    if mode == "np":
        bgm, mix, midi = parse_np_data(path)
    else:
        bgm, mix, midi = parse_serialized_data(path)

    mel = spec2mel(mix)
    audio = mel2audio(mel)
    # 保存mix路
    sf.write(os.path.join(dst_folder_path, dst_name + "_mix" + ".wav"), audio, samplerate)
    # todo 根据需要保存另一路

    if save_midi:
        pkl_to_mid.convert_to_midi_single(midi, os.path.join(dst_folder_path, dst_name) + ".midi")
    return audio


def recover_song(song_folder_path, dst_folder_path, mode="np", limit=None):
    """
    把一首歌的序列化文件（.np/.serialized）转化为音频，并保存

    :param song_folder_path: 歌曲所在路径，路径中存在多个序列化文件
    :param dst_folder_path: 生成音频保存路径
    :param mode: np: numpy格式，serialized: tf保存格式
    :return:
    """
    assert os.path.exists(song_folder_path)
    assert mode in ("np", "serialized")
    if mode == "np":
        file_paths = glob.glob(os.path.join(song_folder_path, "*.npy"))
    else:
        file_paths = glob.glob(os.path.join(song_folder_path, "*.serialized"))

    def sort_fun(path):
        f_name = os.path.split(path)[-1]
        return int(os.path.splitext(f_name)[0])

    file_paths = sorted(file_paths, key=sort_fun)

    # 此处用多进程控制
    for idx, file_path in enumerate(file_paths):
        f_name = os.path.split(file_path)[-1]
        # dst_path = os.path.join(dst_folder_path, os.path.splitext(f_name)[0] + ".wav")
        dst_name = os.path.splitext(f_name)[0]
        recover_seg(file_path, dst_folder_path, dst_name, mode=mode)
        if limit is not None:
            if idx == limit:
                break


def worker(map_dict):
    src_dir = map_dict["src_dir"]
    dst_dir = map_dict["dst_dir"]
    global_id = os.path.split(dst_dir)[-1]
    global_id = global_id.split("_")[-1]
    print(f"{global_id}_start")
    recover_song(src_dir, dst_dir, mode="serialized", limit=10)  # 每首歌最多抽检指定数量片段
    print(f"{global_id}_finish")


def run():
    # 把一个歌曲文件夹中若干片段转化为音频数据
    # song_folder_path = "/deepiano_data/zhaoliang/qingchen_data/npy_negbgm_record/std/bgm_20220906_200604/001"  # 歌曲路径
    # dst_folder_path = "/data1/projects/BGMcloak/tmp_files"  # 歌曲片段保存路径
    # recover_song(song_folder_path, dst_folder_path, mode="np")

    # song_folder_path = "/deepiano_data/zhaoliang/record_data/serialize_bgm_record_delay/std/bgm_record_20220822_20220905_114647/original/20220822export-001"
    # song_folder_path = "/deepiano_data/yuxiaofei/work/data_0718/serialize_changpu_delay_Peilian_SC55_metronome/std/Peilian_xml_SC55_20220811_172927/original/000"
    # song_folder_path = "/deepiano_data/yuxiaofei/work/data_0718/serialize_noise/noise_trans/noise_20220809_170445/1-16k"

    dst_base_root = "/data1/projects/BGMcloak/tmp_files"
    dst_root = os.path.join(dst_base_root, "check_w_wo_bgm_20220916")
    if not os.path.exists(dst_root):
        os.mkdir(dst_root)

    worker_num = 16
    song_dir_list = [
        # # lijun
        # "/deepiano_data/zhaoliang/lijun_data/serialize_bgm_record/noise_trans/bgm_record_20220411_20220915_151310/original/iPhone-001",
        # "/deepiano_data/zhaoliang/lijun_data/serialize_bgm_record/noise_trans/bgm_record_20220411_20220915_151528/original/iPhone-019",
        # "/deepiano_data/zhaoliang/lijun_data/serialize_bgm_record/trans/bgm_specail_20220915_152822/20220629/iPhone-018",
        # "/deepiano_data/zhaoliang/lijun_data/serialize_bgm_record/noise_trans/bgm_record_20220420_20220915_151510/original/iPhone_xs-009",
        # "/deepiano_data/zhaoliang/lijun_data/serialize_bgm_record/noise_trans/bgm_record_peilian_20220915_152711/20220420/iPhone_Xs_021",
        # "/deepiano_data/zhaoliang/lijun_data/serialize_bgm_record/trans/bgm_specail_20220915_152822/20220629/iPhone-029",
        # "/deepiano_data/zhaoliang/lijun_data/serialize_bgm_record/trans/newmusic_20220915_152837/newmusic_0_1_1/8gamemoveyoubody",
        # "/deepiano_data/zhaoliang/lijun_data/serialize_bgm_record/trans/single-note_-20_-5_noised_20220915_153430/20210510/9_85_59_97_25_36_27_25_107_91_103_92_106_81_30_39_100",
        #
        # # record
        # "/deepiano_data/zhaoliang/record_data/serialize_bgm_record/trans/bgm_record_20220822_20220914_192413/original/20220822export-009",
        # "/deepiano_data/zhaoliang/record_data/serialize_bgm_record/noise_trans/bgm_record_20220824_20220914_192817/original/20220824export-012",
        # "/deepiano_data/zhaoliang/record_data/serialize_bgm_record/noise_trans/bgm_record_20220826_20220914_192744/original/20220826export-003",

        # # bgm
        # "/deepiano_data/zhaoliang/qingchen_bgm_data/serialize_negbgm_record/noise_trans/bgm_20220916_113556/ipad5唱谱原版BGM-1",
        # "/deepiano_data/zhaoliang/qingchen_bgm_data/serialize_negbgm_record/noise_trans/bgm_20220916_114908/ipad6无唱谱原版BGM-2",
        # "/deepiano_data/zhaoliang/qingchen_bgm_data/serialize_negbgm_record/trans/bgm_20220916_111516/ipad6唱谱原版BGM-27",
        # "/deepiano_data/zhaoliang/qingchen_bgm_data/serialize_negbgm_record/trans/bgm_20220916_113152/iphonexs111无唱谱原版BGM-7"

        # high note
        "/deepiano_data/zhaoliang/lijun_data/serialize_bgm_record/noise_trans/high-note_-20_-5_noised_20220915_153812/20210513/01-Audio-210507_1029",
        "/deepiano_data/zhaoliang/lijun_data/serialize_bgm_record/trans/high-note_-20_-5_noised_20220915_153536/20210513/01-Audio-210520_1610-single_note",
        # single note
        "/deepiano_data/zhaoliang/lijun_data/serialize_bgm_record/noise_trans/single-note_-20_-5_noised_20220915_153949/20210510/6_45_36_103_104_57_103_51_47_100_95_41_28_89_40_93_46",
        "/deepiano_data/zhaoliang/lijun_data/serialize_bgm_record/trans/single-note_-20_-5_noised_20220915_153430/20210510/7_73_50_23_98_96_47_34_42_63_23_30_75_24_91_81_33"
    ]
    # 目标目录预生成，避免多进程创建目录IO冲突
    convert_map_list = list()  # 转换映射列表
    for dir_id, path in enumerate(song_dir_list):
        assert os.path.exists(path)
        dir_name = os.path.split(path)[-1]  # 源目录名称
        dir_name = dir_name + f"_{dir_id}"  # 增加全局id避免路径冲突
        dst_dir = os.path.join(dst_root, dir_name)  # 目标目录名称
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        convert_map_list.append({
            "src_dir": path,
            "dst_dir": dst_dir,
        })  # 记录源目录，目标目录(path, dst_dir)

    # 保存映射记录
    with open(os.path.join(dst_root, "dir_map.json"), "w") as f:
        json.dump(convert_map_list, f)

    pool = Pool(worker_num)
    ret = pool.map(worker, convert_map_list)
    pool.close()
    pool.join()


if __name__ == '__main__':
    run()
