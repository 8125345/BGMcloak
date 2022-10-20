"""
【早期代码，待删除】构建模型训练数据
"""
import librosa
import numpy as np
import glob
import os
import pickle

# todo 超参待调节
frame_len = 512  # 训练数据的时间片长度

SR = 16000


def spec2pickle(spec, path):
    """
    频谱数据切片并保存为文件
    :param spec_data:
    :return:
    """
    data_dict = dict()
    for i in range(0, spec.shape[0], frame_len):
        start = i
        end = i + frame_len
        slice_spec = spec[start: end, :]
        # 空值补0
        new_matrix = np.pad(slice_spec, ((0, frame_len - slice_spec.shape[0]), (0, 0)), mode='constant',
                            constant_values=0)
        data_dict[i] = new_matrix

    with open(path, "wb") as f:
        pickle.dump(data_dict, f)


def resample(y, src_sr, dst_sr):
    print('RESAMPLING from {} to {}'.format(src_sr, dst_sr))
    if src_sr == dst_sr:
        return y

    if src_sr % dst_sr == 0:
        step = src_sr // dst_sr
        y = y[::step]
        return y
    # Warning, slow
    print('WARNING!!!!!!!!!!!!! SLOW RESAMPLING!!!!!!!!!!!!!!!!!!!!!!!!!!')
    return librosa.resample(y, src_sr, dst_sr)


def wav2spec(fn, sr=SR, hop_length=512, fmin=30.0, n_mels=229, htk=True, spec_log_amplitude=True):
    y, file_sr = librosa.load(fn, mono=True, sr=None)
    y = resample(y, file_sr, sr)
    y = np.concatenate((y, np.zeros(hop_length * 2, dtype=y.dtype)))
    mel = librosa.feature.melspectrogram(
        y,
        sr,
        hop_length=hop_length,
        fmin=fmin,
        n_mels=n_mels,
        htk=htk).astype(np.float32)

    # todo 验证此处转置是否会引发错误
    # Transpose so that the data is in [frame, bins] format.
    spec = mel.T
    if spec_log_amplitude:
        spec = librosa.power_to_db(spec)
    return spec


# def gen_input(wav_filename):
#     spec = wav2spec(wav_filename)
#     for i in range(chunk_padding, spec.shape[0], frames_nopadding):
#         start = i - chunk_padding
#         end = i + frames_nopadding + chunk_padding
#         chunk_spec = spec[start:end]
#         if chunk_spec.shape[0] == chunk_padding * 2 + frames_nopadding:
#             input_item = {
#                 'spec': chunk_spec.flatten()
#             }
#             yield input_item


def process(src_folder, dst_folder):
    audio_paths = glob.glob(os.path.join(src_folder, "*.wav"))
    print(f"audio数量:{len(audio_paths)}")
    for path in audio_paths:
        filename = os.path.split(path)[-1]
        new_path = os.path.join(dst_folder, filename + ".pkl")
        spec = wav2spec(path)
        spec2pickle(spec, new_path)


if __name__ == '__main__':
    # wav_filename = "/deepiano_data/bgm/16k/17-play-铃儿响叮当MMO.wav"
    src_folder = "/deepiano_data/bgm/16k"
    dst_folder = "/data1/dataset/audio/bgm_pkl"
    process(src_folder, dst_folder)

    # spec = librosa.power_to_db(spec)

    # for input_item in gen_input(wav_filename):
    #     # pass
    #     a = 1
    #
    # librosa.feature.inverse.mel_to_audio
    #
    # sf.write('stereo_file.wav', data, samplerate, subtype='PCM_24')
