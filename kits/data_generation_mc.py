import keras
import numpy as np
import SimpleITK as sitk

import kits.AugmentationGenerator as aug
import os


def remove_tumor_segmentation(arr):
    arr[arr > 1] = 1
    return arr


class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_path, ens_path, id_list, batch_size=2, dim=(152, 152, 56), n_channels=1,
                 n_classes=1, shuffle=True, rotation=True, augmentation=False):
        'Initialization'
        self.dim = dim
        self.data_path = data_path
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.rotation = rotation
        self.id_list = id_list
        self.indexes = np.arange(len(self.id_list))
        self.augmentation = augmentation
        self.ens_path = ens_path

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # 32,168,168
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, 1), dtype=np.uint8)
        ENS = np.empty((self.batch_size, *self.dim, 1), dtype=np.uint8)
        flag = np.zeros((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            ID = str.split(ID, '#')
            # side_type = np.random.randint(0, 1)
            # side = 'left' if side_type==0 else 'right'
            img = np.load(self.data_path + '/case_' + str(ID[0]) + '/img_' + ID[1] + '.npy')
            orig_gt = remove_tumor_segmentation(
                np.load(self.data_path + '/case_' + str(ID[0]) + '/segm_' + ID[1] + '.npy'))
            ens_gt = np.load(self.ens_path + '/case_' + str(ID[0]) + '/segm_' + ID[1] + '.npy')
            # print(ID)

            if self.augmentation:
                aug_type = np.random.randint(0, 4)

                aug_img, aug_gt, aug_ens = aug.get_single_image_augmentation_with_ensemble(aug_type,
                                                                                           img,
                                                                                           orig_gt,
                                                                                           ens_gt,
                                                                                           img_no=ID[0])
                X[i] = aug_img
                Y[i] = aug_gt
                ENS[i] = aug_ens

            else:
                X[i, :, :, :, 0] = img  # [10:-10, 10:-10, 10:-10]
                Y[i, :, :, :, 0] = orig_gt  # [10:-10, 10:-10, 10:-10]
                ENS[i, :, :, :, 0] = ens_gt

            flag[i] = np.load(self.ens_path + 'case_' + str(ID[0]) + '/flag_' + ID[1] + '.npy').astype('float16')

            x_t = [X, ENS, flag]
            # del X, ENS, flag

        return x_t, Y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.id_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.id_list[k] for k in indexes]

        # Generate data

        [img, ensemble_pred, flag], Y = self.__data_generation(list_IDs_temp)

        return [img, ensemble_pred, flag], [Y]
