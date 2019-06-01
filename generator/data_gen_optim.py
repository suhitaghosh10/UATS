import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):

    def __init__(self, path, unsupervised_target, unsupervised_weight, supervised_flag, id_list,
                 batch_size=2, dim=(32, 168, 168)):
        'Initialization'
        self.dim = dim
        self.path = path
        self.unsupervised_target = unsupervised_target
        self.supervised_flag = supervised_flag
        self.unsupervised_weight = unsupervised_weight
        self.batch_size = batch_size
        self.id_list = id_list
        self.indexes = np.arange(len(self.id_list))

    def on_epoch_end(self):
        pass

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        img = np.empty((self.batch_size, *self.dim, 1))
        unsup_label = np.zeros((self.batch_size, *self.dim, 5))
        flag = np.zeros((self.batch_size, *self.dim, 1), dtype='int8')
        wt = np.zeros((self.batch_size, *self.dim, 5))

        pz_gt = np.zeros((self.batch_size, *self.dim), dtype='int8')
        cz_gt = np.zeros((self.batch_size, *self.dim), dtype='int8')
        us_gt = np.zeros((self.batch_size, *self.dim), dtype='int8')
        afs_gt = np.zeros((self.batch_size, *self.dim), dtype='int8')
        bg_gt = np.zeros((self.batch_size, *self.dim), dtype='int8')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # img[i] = self.img_array[int(ID)]
            img[i] = np.load(self.path + 'imgs/' + ID + '.npy')
            unsup_label[i] = self.unsupervised_target[int(ID)]
            # gt[i] = self.supervised_label[int(ID)]
            gt = np.load(self.path + 'gt/' + ID + '.npy').astype('int8')
            flag[i] = self.supervised_flag[int(ID)]
            wt[i] = self.unsupervised_weight[int(ID)]

            pz_gt[i] = gt[:, :, :, 0]
            cz_gt[i] = gt[:, :, :, 1]
            us_gt[i] = gt[:, :, :, 2]
            afs_gt[i] = gt[:, :, :, 3]
            bg_gt[i] = gt[:, :, :, 4]

        x_t = [img, unsup_label, flag, wt]
        # x_t = [img, flag]
        y_t = [pz_gt, cz_gt, us_gt, afs_gt, bg_gt]

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
        [img, unsup_label, flag, wt], [pz, cz, us, afs, bg] = self.__data_generation(list_IDs_temp)

        return [img, unsup_label, flag, wt], [pz, cz, us, afs, bg]
