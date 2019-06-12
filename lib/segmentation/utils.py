import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def load_data(data_to_path):
    """load data
    data should be compressed in npz. return arrays
    """
    data = np.load(data_to_path)
    return data['train_x'], data['train_y'], data['test_x'], data['test_y']


def _cal_num_each_train_labal(num_labeled_train, category):
    """calculate the number of data of each class from the whole labeled data size and the number of class"""
    return int(num_labeled_train / len(category))


def split_supervised_train(images, labels, num_labeled_train):
    """get data for supervised training from the whole data.
    This function returns labeled train data and these remainings separetely.
    """
    # list of unique category in labels

    train_labeled_idx = []
    train_list_no = np.arange(0, images.shape[0])
    train_labeled_idx.extend(np.random.choice(train_list_no, num_labeled_train, replace=False))

    # difference set between all-data indices and selected labeled data indices
    diff_set = list(np.setdiff1d(train_list_no, train_labeled_idx))
    print('supervised ex-', train_labeled_idx)
    print('un-supervised ex-', diff_set)
    return {
        'labeled_x': images[train_labeled_idx],
        'labeled_y': labels[train_labeled_idx],
        'unlabeled_x': images[diff_set],
    }


def make_train_test_dataset(train_x, train_y, val_x, val_y, num_labeled_train, num_class,
                            unsupervised_target_init=False):
    """make train dataset and test dataset"""
    ret_dic = {}
    # validation
    ret_dic['val_x'] = val_x
    ret_dic['val_y'] = val_y

    # train
    labeled_set = []
    train_list_no = np.arange(0, train_x.shape[0])
    labeled_set.extend(np.random.choice(train_list_no, num_labeled_train, replace=False))

    # difference set between all-data indices and selected labeled data indices
    unlabeled_set = list(np.setdiff1d(train_list_no, labeled_set))
    train_x = np.concatenate((train_x[labeled_set], train_x[unlabeled_set]), axis=0)

    supervised_label = train_y[labeled_set]
    unsupervised_label = train_y[unlabeled_set]

    num_train_unlabeled = unsupervised_label.shape[0]

    # fill dummy 0 array and the size will corresponds to train dataset at axis 0
    unlabeled_data_label = np.empty_like(unsupervised_label)
    unlabeled_data_label[:] = np.mean(train_y, axis=0)
    supervised_label = np.concatenate((supervised_label, unlabeled_data_label), axis=0)
    num_train_data = supervised_label.shape[0]

    # flag to indicate that supervised(1) or not(0) in train data

    supervised_flag = np.concatenate( (np.ones((num_train_data - num_train_unlabeled, 32, 168, 168,1)), np.zeros((num_train_unlabeled, 32, 168, 168,1))))
    # initialize ensemble prediction label for unsupervised component. It corresponds to matrix Z
    if unsupervised_target_init:
        unsupervised_target = train_y
    else:
        unsupervised_target = np.zeros((num_train_data, 32, 168, 168, num_class))

    # initialize weight of unsupervised loss component
    unsupervised_weight = np.zeros((num_train_data, 32, 168, 168, num_class))

    ret_dic['train_x'] = train_x
    ret_dic['supervised_label'] = supervised_label
    ret_dic['unsupervised_target'] = unsupervised_target
    ret_dic['train_sup_flag'] = supervised_flag
    ret_dic['unsupervised_weight'] = unsupervised_weight

    return ret_dic

def make_train_test_dataset(train_x, train_y, val_x, val_y, num_labeled_train, num_class,
                            unsupervised_target_init=False):
    """make train dataset and test dataset"""
    ret_dic = {}
    # validation
    ret_dic['val_x'] = val_x
    ret_dic['val_y'] = val_y

    # train
    labeled_set = []
    train_list_no = np.arange(0, train_x.shape[0])
    labeled_set.extend(np.random.choice(train_list_no, num_labeled_train, replace=False))

    # difference set between all-data indices and selected labeled data indices
    unlabeled_set = list(np.setdiff1d(train_list_no, labeled_set))
    train_x = np.concatenate((train_x[labeled_set], train_x[unlabeled_set]), axis=0)

    supervised_label = train_y[labeled_set]
    unsupervised_label = train_y[unlabeled_set]

    num_train_unlabeled = unsupervised_label.shape[0]

    # fill dummy 0 array and the size will corresponds to train dataset at axis 0
    unlabeled_data_label = np.empty_like(unsupervised_label)
    unlabeled_data_label[:] = np.mean(train_y, axis=0)
    supervised_label = np.concatenate((supervised_label, unlabeled_data_label), axis=0)
    num_train_data = supervised_label.shape[0]

    # flag to indicate that supervised(1) or not(0) in train data

    supervised_flag = np.concatenate( (np.ones((num_train_data - num_train_unlabeled, 32, 168, 168,1)), np.zeros((num_train_unlabeled, 32, 168, 168,1))))
    # initialize ensemble prediction label for unsupervised component. It corresponds to matrix Z
    if unsupervised_target_init:
        unsupervised_target = train_y
    else:
        unsupervised_target = np.zeros((num_train_data, 32, 168, 168, num_class))

    # initialize weight of unsupervised loss component
    unsupervised_weight = np.zeros((num_train_data, 32, 168, 168, num_class))

    ret_dic['train_x'] = train_x
    ret_dic['supervised_label'] = supervised_label
    ret_dic['unsupervised_target'] = unsupervised_target
    ret_dic['train_sup_flag'] = supervised_flag
    ret_dic['unsupervised_weight'] = unsupervised_weight

    return ret_dic


def make_dataset(train_x, train_y, train_ux, train_uy, val_x, val_y, num_class):
    """make train dataset and test dataset"""
    ret_dic = {}
    # validation
    ret_dic['val_x'] = val_x
    ret_dic['val_y'] = val_y.astype('int8')

    # train
    num_labeled_train = train_x.shape[0]
    num_un_labeled_train = train_ux.shape[0]
    total_train_num = num_labeled_train + num_un_labeled_train

    imgs = np.concatenate((train_x, train_ux), axis=0)

    supervised_label = np.concatenate((train_y, train_uy), axis=0)

    # flag to indicate that supervised(1) or not(0) in train data

    supervised_flag = np.concatenate(
        (np.ones((num_labeled_train, 32, 168, 168, 1)), np.zeros((num_un_labeled_train, 32, 168, 168, 1))))
    unsupervised_target = np.concatenate((train_y, train_uy))

    # initialize weight of unsupervised loss component
    unsupervised_weight = np.zeros((total_train_num, 32, 168, 168, num_class))

    ret_dic['train_x'] = imgs
    ret_dic['supervised_label'] = supervised_label.astype('int8')
    ret_dic['unsupervised_target'] = unsupervised_target.astype('float32')
    ret_dic['train_sup_flag'] = supervised_flag.astype('int8')
    ret_dic['unsupervised_weight'] = unsupervised_weight.astype('float32')
    del imgs, supervised_label, unsupervised_target, supervised_flag, unsupervised_weight

    return ret_dic


def get_complete_array(folder_path, dtype=None):
    files = os.listdir(folder_path)
    total_arr = None
    for idx in np.arange(len(files)):
        if (idx == 0):
            arr = np.load(folder_path + str(idx) + '.npy')
            if (dtype is None):
                total_arr = np.zeros((len(files), *arr.shape))
            else:
                total_arr = np.zeros((len(files), *arr.shape), dtype=dtype)
            total_arr[0] = arr
        else:
            total_arr[idx] = np.load(folder_path + str(idx) + '.npy')
    return total_arr


def get_array(folder_path, start, end, dtype=None):
    arr = np.load(folder_path + '/0.npy')
    if dtype is None:
        total_arr = np.zeros((end - start, *arr.shape))
    else:
        total_arr = np.zeros((end - start, *arr.shape), dtype=dtype)
    for idx in np.arange(start, end):
        arr_idx = idx - start
        total_arr[arr_idx] = np.load(folder_path + '/' + str(idx) + '.npy')

    return total_arr


def save_array(path, arr, start, end):
    for idx in np.arange(start, end):
        arr_idx = idx - start
        np.save(path + str(idx) + '.npy', arr[arr_idx])




def data_augmentation_tempen(inputs, trans_range):
    """data augmentation by random translation and horizonal flip.
    This implementation refers to the author's implementation of the paper.
    """
    temp_lst = []
    for img in inputs:
        if np.random.uniform() > 0.5:
            # horizonal flip. NHWC
            img = img[:, ::-1, :]

        p0 = np.random.randint(-trans_range, trans_range + 1) + trans_range
        p1 = np.random.randint(-trans_range, trans_range + 1) + trans_range

        img = img[p0:p0 + 32, p1:p1 + 32, :]
        temp_lst.append(img)

    return np.array(temp_lst)


def to_categorical_onehot(label, num_class):
    """transform categorical labels to one-hot vectors"""
    return np.identity(num_class)[label]


def normalize_images(*arrays):
    """normalize all input arrays by dividing 255"""
    return [arr / 255 for arr in arrays]


def whiten_zca(x_train, x_test):
    """whiten train and test data by zca whitening"""
    zca_gen = ImageDataGenerator(zca_whitening=True)
    zca_gen.fit(x_train)

    g_train = zca_gen.flow(x_train, batch_size=len(x_train), shuffle=False)
    g_test = zca_gen.flow(x_test, batch_size=len(x_test), shuffle=False)

    x_train = g_train.next()
    x_test = g_test.next()

    return x_train, x_test


if __name__ == '__main__':
    a = get_array('/home/suhita/zonals/data/validation/gt/', start=0, end=20, data_type='int8')
    b = np.load('/home/suhita/zonals/data/validation/valArray_GT_fold1.npy')
    print(np.count_nonzero(a - b))
