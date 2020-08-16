import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):

    def __init__(self, path, id_list, batch_size, dim, n_channels, n_classes, shuffle=True, rotation=True):
        'Initialization'
        self.dim = dim
        self.img_path = path+'/imgs/'
        self.gt_path = path+'/gt/'
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.rotation = rotation
        self.id_list = id_list
        self.indexes = np.arange(len(self.id_list))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # 32,168,168
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y1 = np.empty((self.batch_size, *self.dim), dtype=np.uint8)
        y2 = np.empty((self.batch_size, *self.dim), dtype=np.uint8)
        y3 = np.empty((self.batch_size, *self.dim), dtype=np.uint8)
        y4 = np.empty((self.batch_size, *self.dim), dtype=np.uint8)
        y5 = np.empty((self.batch_size, *self.dim), dtype=np.uint8)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            X[i, :, :, :, :] = np.load(self.img_path + str(ID) + '.npy')
            aug_gt = np.load(self.gt_path + str(ID) + '.npy')

            y1[i, :, :, :] = aug_gt[:, :, :, 0]
            y2[i, :, :, :] = aug_gt[:, :, :, 1]
            y3[i, :, :, :] = aug_gt[:, :, :, 2]
            y4[i, :, :, :] = aug_gt[:, :, :, 3]
            y5[i, :, :, :] = aug_gt[:, :, :, 4]

        return X, [y1, y2, y3, y4, y5]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.id_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print('\n')
        # print(indexes)

        # Find list of IDs
        list_IDs_temp = [self.id_list[k] for k in indexes]

        # Generate data
        X, [y1, y2, y3, y4, y5] = self.__data_generation(list_IDs_temp)

        return X, [y1, y2, y3, y4, y5]
