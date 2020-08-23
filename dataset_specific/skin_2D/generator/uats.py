import os
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from dataset_specific.skin_2D.utils.preprocess import augment_image

class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_path, ens_path, id_list, batch_size=8, dim=(192, 256), n_channels=3,
                 n_classes=1, shuffle=True, rotation=True, is_augmented=True, labelled_num=None):
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
        self.is_augmented = is_augmented
        self.datagen_img = ImageDataGenerator(fill_mode='constant', cval=0, interpolation_order=1)
        self.datagen_GT = ImageDataGenerator(fill_mode='constant', cval=0, interpolation_order=0)
        self.datagen_ens = ImageDataGenerator(fill_mode='constant', cval=0, interpolation_order=0)
        self.datagen_flag = ImageDataGenerator(fill_mode='constant', cval=0, interpolation_order=0)

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # 32,168,168
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=float)
        Y = np.empty((self.batch_size, *self.dim, 2), dtype=np.uint8)
        ENS = np.empty((self.batch_size, *self.dim, 2))
        FLAG = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #print(ID)
            x = np.load(os.path.join(self.data_path, 'imgs', str(ID) + '.npy'))
            y = np.load(os.path.join(self.data_path, 'gt', str(ID) + '.npy'))
            ens = np.load(os.path.join(self.ens_path, 'ens_gt', str(ID) + '.npy'))
            flag = np.load(os.path.join(self.ens_path, 'flag', str(ID) + '.npy')).reshape((self.dim[0], self.dim[1],1))

            if self.is_augmented:
                aug_type = np.random.randint(0, 6)
                X[i, :, :, :] = augment_image(x, aug_type, self.datagen_img)
                Y[i] = augment_image(y, aug_type, self.datagen_GT)
                ENS[i] = augment_image(ens, aug_type, self.datagen_ens)
                FLAG[i] = augment_image(flag, aug_type, self.datagen_flag)[:, :, 0]
            else:
                X[i] = x
                Y[i] = y
                ENS[i] = ens
                FLAG[i] = flag

        X_ARR = [X, ENS, FLAG]

        return X_ARR, [Y[:, :, :, 0], Y[:, :, :, 1]]

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

        return [img, ensemble_pred, flag], Y
