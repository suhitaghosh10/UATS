import numpy as np


def get_multi_class_arr(arr, n_classes=3):
    size = arr.shape
    out_arr = np.zeros([size[0], size[1], size[2], n_classes])

    for i in range(n_classes):
        arr_temp = arr.copy()
        out_arr[:, :, :, i] = np.where(arr_temp == i, 1, 0)
        del arr_temp
    return out_arr
