import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):

    def __init__(self, img_array, unsupervised_target, supervised_label, supervised_flag, unsupervised_weight, id_list,
                 batch_size=2, dim=(32, 168, 168), n_channels=1,
                 n_classes=10, shuffle=True, rotation=True):
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
        self.rotation = rotation
        self.indexes = np.arange(len(self.id_list))
        # if self.shuffle == True:
        #    np.random.shuffle(self.indexes)
        print('data gen-', self.indexes)
        # self.on_epoch_end()

    def on_epoch_end(self):
        pass

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        img = np.zeros((self.batch_size, *self.dim, 1))
        unsup_label = np.zeros((self.batch_size, *self.dim, 5))
        gt = np.zeros((self.batch_size, *self.dim, 5))
        flag = np.zeros((self.batch_size, *self.dim, 1))
        wt = np.zeros((self.batch_size, *self.dim, 5))

        pz = np.zeros((self.batch_size, *self.dim, 4))
        cz = np.zeros((self.batch_size, *self.dim, 4))
        us = np.zeros((self.batch_size, *self.dim, 4))
        afs = np.zeros((self.batch_size, *self.dim, 4))
        bg = np.zeros((self.batch_size, *self.dim, 4))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img[i] = self.img_array[int(ID)]
            unsup_label[i] = self.unsupervised_target[int(ID)]
            gt[i] = self.supervised_label[int(ID)]
            flag[i] = self.supervised_flag[int(ID)]
            wt[i] = self.unsupervised_weight[int(ID)]

            pz[i] = np.stack((self.unsupervised_target[int(ID), :, :, :, 0], self.supervised_label[int(ID), :, :, :, 0],
                              self.supervised_flag[int(ID), :, :, :, 0], self.unsupervised_weight[int(ID), :, :, :, 0]),
                             axis=-1)

            cz[i] = np.stack((self.unsupervised_target[int(ID), :, :, :, 1], self.supervised_label[int(ID), :, :, :, 1],
                              self.supervised_flag[int(ID), :, :, :, 0], self.unsupervised_weight[int(ID), :, :, :, 1]),
                             axis=-1)

            us[i] = np.stack((self.unsupervised_target[int(ID), :, :, :, 2], self.supervised_label[int(ID), :, :, :, 2],
                              self.supervised_flag[int(ID), :, :, :, 0], self.unsupervised_weight[int(ID), :, :, :, 2]),
                             axis=-1)

            afs[i] = np.stack(
                (self.unsupervised_target[int(ID), :, :, :, 3], self.supervised_label[int(ID), :, :, :, 3],
                 self.supervised_flag[int(ID), :, :, :, 0], self.unsupervised_weight[int(ID), :, :, :, 3]),
                axis=-1)

            bg[i] = np.stack((self.unsupervised_target[int(ID), :, :, :, 4], self.supervised_label[int(ID), :, :, :, 4],
                              self.supervised_flag[int(ID), :, :, :, 0], self.unsupervised_weight[int(ID), :, :, :, 4]),
                             axis=-1)

        y_t = [pz, cz, us, afs, bg]
        x_t = [img, unsup_label, gt, flag, wt]

        return x_t, y_t
        # return X, [weight1, weight2, weight3, weight4, weight5]
        # return X, [y1, y2, y3, y4, y5], [img_gt1, img_gt1, img_gt1, img_gt1, img_gt1]

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
        [x1, x2, x3, x4], [pz, cz, us, afs, bg] = self.__data_generation(list_IDs_temp)

        return [x1, x2, x3, x4], [pz, cz, us, afs, bg]