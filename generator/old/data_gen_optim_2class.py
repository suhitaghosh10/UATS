import keras

from generator.AugmentationGenerator import *

NPY = '.npy'


class DataGenerator(keras.utils.Sequence):
    def __init__(self, imgs_path, gt_path, id_list, batch_size=2,
                 dim=(32, 168, 168)):
        'Initialization'
        self.dim = dim
        self.imgs_path = imgs_path
        self.gt_path = gt_path
        self.batch_size = batch_size
        self.id_list = id_list
        self.indexes = np.arange(len(self.id_list))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        img = np.empty((self.batch_size, *self.dim, 1))
        gt = np.empty((self.batch_size, *self.dim), dtype='int8')
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img[i] = np.load(self.imgs_path + ID + NPY)
            gt[i] = np.load(self.gt_path + ID + NPY).astype('int8')[:, :, :, 3]

        return img, gt

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
        img, gt = self.__data_generation(list_IDs_temp)

        return img, gt
