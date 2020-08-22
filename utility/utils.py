import os
import numpy as np
import tensorflow.keras.backend as K
from utility.config import get_metadata
from utility.constants import *
import shutil


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


def makedir(dirpath, delete_existing=False):

    if delete_existing and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
        os.makedirs(dirpath)
    elif not os.path.isdir(dirpath):
        os.makedirs(dirpath)



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


# def get_val_data(data_path):
#     val_fold = os.listdir(data_path[:-7] + VAL_IMGS_PATH)
#     num_val_data = len(val_fold)
#     val_supervised_flag = np.ones((num_val_data, PROSTATE_DIM[0], PROSTATE_DIM[1], PROSTATE_DIM[2]), dtype='int8')
#     val_img_arr = np.zeros((num_val_data, PROSTATE_DIM[0], PROSTATE_DIM[1], PROSTATE_DIM[2], 1), dtype=float)
#     val_GT_arr = np.zeros((num_val_data, PROSTATE_DIM[0], PROSTATE_DIM[1], PROSTATE_DIM[2], PROSTATE_NR_CLASS),
#                           dtype=float)
#     for i in np.arange(num_val_data):
#         val_img_arr[i] = np.load(data_path[:-7] + VAL_IMGS_PATH + str(i) + NPY)
#         val_GT_arr[i] = np.load(data_path[:-7] + VAL_GT_PATH + str(i) + NPY)
#     x_val = [val_img_arr, val_GT_arr, val_supervised_flag]
#     y_val = [val_GT_arr[:, :, :, :, 0], val_GT_arr[:, :, :, :, 1],
#              val_GT_arr[:, :, :, :, 2], val_GT_arr[:, :, :, :, 3],
#              val_GT_arr[:, :, :, :, 4]]
#     return x_val, y_val


def get_uats_val_data(data_path, dim, nr_class, nr_channels):
    data_path = data_path[:-6]
    val_fold = os.listdir(data_path + VAL_IMGS_PATH)
    num_val_data = len(val_fold)
    if len(dim) == 2:
        val_supervised_flag = np.ones((num_val_data, dim[0], dim[1]), dtype='int64')
        val_img_arr = np.zeros((num_val_data, dim[0], dim[1], nr_channels), dtype=float)
        val_GT_arr = np.zeros((num_val_data, dim[0], dim[1], nr_class), dtype=float)
    else:
        val_supervised_flag = np.ones((num_val_data, dim[0], dim[1], dim[2]), dtype='int64')
        val_img_arr = np.zeros((num_val_data, dim[0], dim[1], dim[2], nr_channels), dtype=float)
        val_GT_arr = np.zeros((num_val_data, dim[0], dim[1], dim[2], nr_class), dtype=float)

    for i in np.arange(num_val_data):
        val_img_arr[i] = np.load(data_path + VAL_IMGS_PATH + str(i) + NPY)
        val_GT_arr[i] = np.load(data_path + VAL_GT_PATH + str(i) + NPY)

    x_val = [val_img_arr, val_GT_arr, val_supervised_flag]
    y_val = []

    for idx in range(nr_class):
        if len(dim) == 3:
            y_val.append(val_GT_arr[:, :, :, :, idx])
        else:
            y_val.append(val_GT_arr[:, :, :, idx])
    return x_val, y_val


def get_temporal_val_data(data_path, dim, nr_class, nr_channels):
    data_path = data_path[:-6]
    val_fold = os.listdir(data_path + VAL_IMGS_PATH)
    num_val_data = len(val_fold)
    if len(dim) == 2:
        val_img_arr = np.zeros((num_val_data, dim[0], dim[1], nr_channels), dtype=float)
        val_GT_arr = np.zeros((num_val_data, dim[0], dim[1], nr_class), dtype=float)
    else:
        val_img_arr = np.zeros((num_val_data, dim[0], dim[1], dim[2], nr_channels), dtype=float)
        val_GT_arr = np.zeros((num_val_data, dim[0], dim[1], dim[2], nr_class), dtype=float)

    for i in np.arange(num_val_data):
        val_img_arr[i] = np.load(data_path + VAL_IMGS_PATH + str(i) + NPY)
        val_GT_arr[i] = np.load(data_path + VAL_GT_PATH + str(i) + NPY)

    x_val = [val_img_arr, val_GT_arr]
    y_val = []
    for idx in range(nr_class):
        y_val.append(val_GT_arr[:, :, :, :, idx])
    return x_val, y_val

def get_supervised_val_data(data_path, dim, nr_class, nr_channels):
    data_path = data_path[:-6]
    val_fold = os.listdir(data_path + VAL_IMGS_PATH)
    num_val_data = len(val_fold)
    if len(dim) == 2:
        val_img_arr = np.zeros((num_val_data, dim[0], dim[1], nr_channels), dtype=float)
        val_GT_arr = np.zeros((num_val_data, dim[0], dim[1], nr_class), dtype=float)
    else:
        val_img_arr = np.zeros((num_val_data, dim[0], dim[1], dim[2], nr_channels), dtype=float)
        val_GT_arr = np.zeros((num_val_data, dim[0], dim[1], dim[2], nr_class), dtype=float)

    for i in np.arange(num_val_data):
        val_img_arr[i] = np.load(data_path + VAL_IMGS_PATH + str(i) + NPY)
        val_GT_arr[i] = np.load(data_path + VAL_GT_PATH + str(i) + NPY)

    x_val = [val_img_arr]
    y_val = []
    for idx in range(nr_class):
        if len(dim)==3:
            y_val.append(val_GT_arr[:, :, :, :, idx])
        else:
            y_val.append(val_GT_arr[:, :, :, idx])

    return x_val, y_val


def get_uats_data_generator(dataset_name, data_path, ens_path, num_train, num_train_labelled, batch_size,
                            is_augmented=True):
    if dataset_name == PROSTATE_DATASET_NAME:
        train_id_list = np.arange(num_train)
        np.random.shuffle(train_id_list)
        print(train_id_list[0:10])

        from dataset_specific.prostate.generator.uats import DataGenerator as train_gen
        return train_gen(data_path,
                         ens_path,
                         train_id_list,
                         batch_size=batch_size,
                         labelled_num=num_train_labelled,
                         is_augmented=is_augmented)

    elif dataset_name == SKIN_DATASET_NAME:
        train_id_list = np.arange(num_train)
        np.random.shuffle(train_id_list)
        print(train_id_list[0:10])

        from dataset_specific.skin_2D.generator.uats import DataGenerator as train_gen
        return train_gen(data_path,
                         ens_path,
                         train_id_list,
                         batch_size=batch_size,
                         labelled_num=num_train_labelled,
                         is_augmented=is_augmented)

def get_supervised_data_generator(dataset_name, data_path, num_train, is_augmented=True):
    if dataset_name == PROSTATE_DATASET_NAME:
        metadata = get_metadata(dataset_name)
        train_id_list = np.arange(num_train)
        np.random.shuffle(train_id_list)
        print(train_id_list[0:10])

        from dataset_specific.prostate.generator.baseline import DataGenerator as train_gen
        return train_gen(data_path,
                         train_id_list,
                         batch_size=metadata[m_batch_size],
                         dim=metadata[m_dim],
                         is_augmented=is_augmented)

    elif dataset_name == SKIN_DATASET_NAME:
        metadata = get_metadata(dataset_name)
        train_id_list = np.arange(num_train)
        np.random.shuffle(train_id_list)
        print(train_id_list[0:10])

        from dataset_specific.skin_2D.generator.baseline import DataGenerator as train_gen
        return train_gen(data_path,
                         train_id_list,
                         batch_size=metadata[m_batch_size],
                         dim=metadata[m_dim],
                         is_augmented=is_augmented)


def get_temporal_data_generator(dataset_name, data_path, ens_path, num_train, num_train_labelled, batch_size,
                                is_augmented=True):
    if dataset_name == PROSTATE_DATASET_NAME:
        train_id_list = np.arange(num_train)
        np.random.shuffle(train_id_list)
        print(train_id_list[0:10])

        if is_augmented:
            from dataset_specific.prostate.generator.temporal import DataGenerator as train_gen
        else:
            from dataset_specific.prostate.generator.temporal import DataGenerator as train_gen
        return train_gen(data_path,
                         ens_path,
                         train_id_list,
                         batch_size=batch_size,
                         labelled_num=num_train_labelled)
