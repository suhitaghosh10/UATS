import os

import keras
import numpy as np

import dataset_specific.hippocampus.AugmentationGenerator as aug


class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_path, ens_path, id_list, batch_size=4, dim=(48, 64, 48), n_channels=1,
                 n_classes=3, shuffle=True, rotation=True, augmentation=True):
        'Initialization'
        self.dim = dim
        self.data_path = data_path
        self.ens_path = ens_path
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
        Y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=np.uint8)
        ens = np.empty((self.batch_size, *self.dim, self.n_classes))
        flag = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            orig_gt = np.load(os.path.join(self.data_path, 'GT', str(ID) + '.npy'))
            orig_ens = np.load(os.path.join(self.ens_path, 'ens_gt', str(ID) + '.npy'))
            flag[i] = np.load(os.path.join(self.ens_path, 'flag', str(ID) + '.npy'))

            if self.augmentation:
                aug_type = np.random.randint(0, 5)
                # print(ID, orig_gt.shape)
                X[i], Y[i], ens[i] = aug.get_single_image_augmentation_with_ens(aug_type,
                                                                                np.load(
                                                                                    os.path.join(self.data_path, 'imgs',
                                                                                                 str(ID) + '.npy')),
                                                                                orig_gt,
                                                                                orig_ens,
                                                                                ID)

            else:
                X[i] = np.load(os.path.join(self.data_path, ID))
                Y[i] = orig_gt
                ens[i] = orig_ens

        return [X, ens, flag], [Y[:, :, :, :, 0], Y[:, :, :, :, 1], Y[:, :, :, :, 2]]

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

        [X, ens, flag], Y = self.__data_generation(list_IDs_temp)

        return [X, ens, flag], Y
