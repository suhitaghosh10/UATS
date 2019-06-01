import keras
import numpy as np

class DataGenerator(keras.utils.Sequence):

    def __init__(self, img_array, unsupervised_target, supervised_label, supervised_flag, unsupervised_weight, id_list,
                 batch_size=2, dim=(32, 168, 168), n_channels=1,
                 n_classes=5, shuffle=True):
        'Initialization'
        self.dim = dim
        self.img_array = img_array
        self.unsupervised_target = unsupervised_target
        self.supervised_label = supervised_label
        self.supervised_flag = supervised_flag
        self.unsupervised_weight = unsupervised_weight
        self.batch_size = batch_size
        self.id_list = id_list
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.id_list))
        # print('data gen-', self.indexes)

    def on_epoch_end(self):
        pass

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        img = np.zeros((self.batch_size, *self.dim, 1))
        unsup_label = np.zeros((self.batch_size, *self.dim, 5))
        gt = np.zeros((self.batch_size, *self.dim, 5), dtype=np.int)
        flag = np.zeros((self.batch_size, *self.dim, 1), dtype=np.int)
        wt = np.zeros((self.batch_size, *self.dim, 5))

        pz = np.zeros((self.batch_size, *self.dim))
        cz = np.zeros((self.batch_size, *self.dim))
        us = np.zeros((self.batch_size, *self.dim))
        afs = np.zeros((self.batch_size, *self.dim))
        bg = np.zeros((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img[i] = self.img_array[int(ID)]
            unsup_label[i] = self.unsupervised_target[int(ID)]
            gt[i] = self.supervised_label[int(ID)]
            flag[i] = self.supervised_flag[int(ID)]
            wt[i] = self.unsupervised_weight[int(ID)]

            pz[i] = self.supervised_label[int(ID), :, :, :, 0]
            cz[i] = self.supervised_label[int(ID), :, :, :, 1]
            us[i] = self.supervised_label[int(ID), :, :, :, 2]
            afs[i] = self.supervised_label[int(ID), :, :, :, 3]
            bg[i] = self.supervised_label[int(ID), :, :, :, 4]

        x_t = [img, unsup_label, gt, flag, wt]
        y_t = [pz, cz, us, afs, bg]


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
        [img, unsup_label, gt, flag, wt], [pz, cz, us, afs, bg] = self.__data_generation(list_IDs_temp)

        return [img, unsup_label, gt, flag, wt], [pz, cz, us, afs, bg]
