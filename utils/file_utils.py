import os


# import audio_tools


def traverse_fun(path, fun):
    """
    对文件执行方法
    :param path:
    :param fun:
    :return:
    """
    for curDir, dirs, files in os.walk(path):
        for file in files:
            fun(os.path.join(curDir, file))


def get_folders(path):
    """
    获取全部文件夹
    :param path:
    :return:
    """

    folders = list()
    for curDir, dirs, files in os.walk(path):
        for dir in dirs:
            folders.append(os.path.join(curDir, dir))
    return folders
