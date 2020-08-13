import os

import keras
import numpy as np

import dataset_specific.kits.AugmentationGenerator as aug


def remove_tumor_segmentation(arr):
    arr[arr > 1] = 1
    return arr


class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_path, id_list, batch_size=2, dim=(32, 168, 168), n_channels=1,
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

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # 32,168,168
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, 1), dtype=np.uint8)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            segm_file_name = ID.replace('img', 'segm')
            orig_gt = remove_tumor_segmentation(np.load(os.path.join(self.data_path, segm_file_name)))
            # Y[i, :, :, :, 0] = remove_tumor_segmentation(np.load(os.path.join(self.data_path, segm_file_name)))#[10:-10, 10:-10, 2:-2]

            # sitk.WriteImage(sitk.GetImageFromArray(X[i, :, :, :, 0]), 'img.nrrd')
            # sitk.WriteImage(sitk.GetImageFromArray(Y[i, :, :, :, 0]), 'segm.nrrd')

            if self.augmentation:
                aug_type = np.random.randint(0, 4)

                X[i], aug_gt = aug.get_single_image_augmentation(aug_type,
                                                                 np.load(os.path.join(self.data_path, ID)),
                                                                 orig_gt, img_no=ID)
                Y[i] = aug_gt

            else:
                X[i, :, :, :, 0] = np.load(os.path.join(self.data_path, ID))  # [10:-10, 10:-10, 10:-10]
                Y[i, :, :, :, 0] = orig_gt  # [10:-10, 10:-10, 10:-10]

        return X, Y

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

        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y
