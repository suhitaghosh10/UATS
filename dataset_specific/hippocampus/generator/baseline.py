import keras
import numpy as np

from dataset_specific.prostate.generator.AugmentationGenerator import get_single_image_augmentation
from utility.constants import NPY


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_path, id_list, batch_size, dim, is_augmented=True):
        'Initialization'
        self.dim = dim
        self.data_path = data_path
        self.batch_size = batch_size
        self.id_list = id_list
        self.indexes = np.arange(len(self.id_list))
        self.is_augmented = is_augmented

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        img = np.empty((self.batch_size, *self.dim, 1))

        z1_gt = np.zeros((self.batch_size, *self.dim))
        z2_gt = np.zeros((self.batch_size, *self.dim))
        bg_gt = np.zeros((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            if self.is_augmented:
                aug_type = np.random.randint(0, 5)
                img[i, :, :, :, :], gt= get_single_image_augmentation(
                aug_type,
                np.load(self.data_path + '/imgs/' + str(ID) + NPY),
                np.load(self.data_path + '/gt/' + str(ID) + NPY))
            else:
                img[i, :, :, :, :] = np.load(self.data_path + '/imgs/' + str(ID) + NPY)
                gt = np.load(self.data_path + '/gt/' + str(ID) + NPY)

            z1_gt[i] = gt[:, :, :, 0]
            z2_gt[i] = gt[:, :, :, 1]
            bg_gt[i] = gt[:, :, :, 2]

        x_t = img
        y_t = [z1_gt, z2_gt,  bg_gt]
        del gt, img

        return x_t, y_t

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.id_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print('\n')
        # print(indexes)

        # Find list of IDs
        list_IDs_temp = [self.id_list[k] for k in indexes]

        # Generate data
        img, [z1_gt, z2_gt,  bg_gt] = self.__data_generation(list_IDs_temp)

        return img, [z1_gt, z2_gt,  bg_gt]
