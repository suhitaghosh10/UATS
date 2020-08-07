import os
import shutil

import keras.backend as K
import numpy as np

from utility.constants import *


def shall_save(cur_val, prev_val):
    flag_save = False
    val_save = prev_val

    if cur_val > prev_val:
        flag_save = True
        val_save = cur_val

    return flag_save, val_save


def cleanup(path):
    shutil.rmtree(path)
    K.clear_session()


def makedir(dirpath):
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    try:
        os.makedirs(dirpath)
    except OSError:
        # [Errno 17] File exists
        pass


def get_array(folder_path, start, end, dtype=None):
    arr = np.load(folder_path + '/0.npy')
    if dtype is None:
        total_arr = np.zeros((end - start, *arr.shape))
    else:
        total_arr = np.zeros((end - start, *arr.shape), dtype=dtype)
    for idx in np.arange(start, end):
        arr_idx = idx - start
        total_arr[arr_idx] = np.load(os.path.join(folder_path, str(idx) + NPY))

    return total_arr


def save_array(path, arr, start, end):
    for idx in np.arange(start, end):
        arr_idx = idx - start
        np.save(os.path.join(path, str(idx) + NPY), arr[arr_idx])
