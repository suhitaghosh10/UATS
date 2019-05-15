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

    return {
        'labeled_x': images[train_labeled_idx],
        'labeled_y': labels[train_labeled_idx],
        'unlabeled_x': images[diff_set],
    }


def make_train_test_dataset(inp_dic, num_class):
    """make train dataset and test dataset"""
    train_x = np.concatenate((inp_dic['labeled_x'], inp_dic['unlabeled_x']), axis=0)

    # transform categorical labels to one-hot vectors
    supervised_label = inp_dic['labeled_y']
    supervised_label_pz = inp_dic['labeled_y'][:, :, :, :, 0, np.newaxis]
    supervised_label_cz = inp_dic['labeled_y'][:, :, :, :, 1, np.newaxis]
    supervised_label_us = inp_dic['labeled_y'][:, :, :, :, 2, np.newaxis]
    supervised_label_afs = inp_dic['labeled_y'][:, :, :, :, 3, np.newaxis]
    supervised_label_bg = inp_dic['labeled_y'][:, :, :, :, 4, np.newaxis]

    test_y = inp_dic['test_y']
    test_y_pz = inp_dic['test_y'][:, :, :, :, 0, np.newaxis]
    test_y_cz = inp_dic['test_y'][:, :, :, :, 1, np.newaxis]
    test_y_us = inp_dic['test_y'][:, :, :, :, 2, np.newaxis]
    test_y_afs = inp_dic['test_y'][:, :, :, :, 3, np.newaxis]
    test_y_bg = inp_dic['test_y'][:, :, :, :, 4, np.newaxis]


    num_train_unlabeled = inp_dic['unlabeled_x'].shape[0]

    # fill dummy 0 array and the size will corresponds to train dataset at axis 0
    unlabeled_data_label = np.zeros((num_train_unlabeled, 32, 168, 168, num_class))
    unlabeled_data_label_pz = np.zeros((num_train_unlabeled, 32, 168, 168, 1))
    unlabeled_data_label_cz = np.zeros((num_train_unlabeled, 32, 168, 168, 1))
    unlabeled_data_label_us = np.zeros((num_train_unlabeled, 32, 168, 168, 1))
    unlabeled_data_label_afs = np.zeros((num_train_unlabeled, 32, 168, 168, 1))
    unlabeled_data_label_bg = np.zeros((num_train_unlabeled, 32, 168, 168, 1))

    supervised_label = np.concatenate((supervised_label, unlabeled_data_label), axis=0)
    supervised_label_pz = np.concatenate((supervised_label_pz, unlabeled_data_label_pz), axis=0)
    supervised_label_cz = np.concatenate((supervised_label_cz, unlabeled_data_label_cz), axis=0)
    supervised_label_us = np.concatenate((supervised_label_us, unlabeled_data_label_us), axis=0)
    supervised_label_afs = np.concatenate((supervised_label_afs, unlabeled_data_label_afs), axis=0)
    supervised_label_bg = np.concatenate((supervised_label_bg, unlabeled_data_label_bg), axis=0)
    num_train_data = supervised_label_pz.shape[0]

    # flag to indicate that supervised(1) or not(0) in train data

    supervised_flag = np.concatenate( (np.ones((num_train_data - num_train_unlabeled, 32, 168, 168,1)), np.zeros((num_train_unlabeled, 32, 168, 168,1))))
    # initialize ensemble prediction label for unsupervised component. It corresponds to matrix Z
    unsupervised_target = np.zeros((num_train_data, 32, 168, 168, num_class))
    unsupervised_target_pz = np.zeros((num_train_data, 32, 168, 168, 1))
    unsupervised_target_cz = np.zeros((num_train_data, 32, 168, 168, 1))
    unsupervised_target_us = np.zeros((num_train_data, 32, 168, 168, 1))
    unsupervised_target_afs = np.zeros((num_train_data, 32, 168, 168, 1))
    unsupervised_target_bg = np.zeros((num_train_data, 32, 168, 168, 1))

    # initialize weight of unsupervised loss component
    unsupervised_weight = np.zeros((num_train_data, 32, 168, 168, num_class))
    unsupervised_weight_pz = np.zeros((num_train_data, 32, 168, 168, 1))
    unsupervised_weight_cz = np.zeros((num_train_data, 32, 168, 168, 1))
    unsupervised_weight_us = np.zeros((num_train_data, 32, 168, 168, 1))
    unsupervised_weight_afs = np.zeros((num_train_data, 32, 168, 168, 1))
    unsupervised_weight_bg = np.zeros((num_train_data, 32, 168, 168, 1))

    del inp_dic['labeled_x'], inp_dic['labeled_y'], inp_dic['unlabeled_x']
    inp_dic['train_x'] = train_x
    inp_dic['supervised_label_y'] = [supervised_label_pz, supervised_label_cz, supervised_label_us,
                                     supervised_label_afs,
                                     supervised_label_bg]
    inp_dic['supervised_label_x'] = supervised_label

    inp_dic['unsupervised_target_y'] = [unsupervised_target_pz, unsupervised_target_cz, unsupervised_target_us,
                                      unsupervised_target_afs, unsupervised_target_bg]
    inp_dic['unsupervised_target_x'] = unsupervised_target

    inp_dic['train_sup_flag'] = supervised_flag

    inp_dic['unsupervised_weight_y'] = [unsupervised_weight_pz, unsupervised_weight_cz, unsupervised_weight_us,
                                      unsupervised_weight_afs, unsupervised_weight_bg]
    inp_dic['unsupervised_weight_x'] = unsupervised_weight

    inp_dic['test_y'] = [test_y_pz, test_y_cz, test_y_us, test_y_afs, test_y_bg]

    return inp_dic


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
