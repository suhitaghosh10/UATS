import keras

NPY = '.npy'


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_path, ensemble_path, id_list, batch_size=2,
                 dim=(32, 168, 168), labelled_num=58):
        'Initialization'
        self.dim = dim
        self.data_path = data_path
        self.ensemble_path = ensemble_path
        self.batch_size = batch_size
        self.id_list = id_list
        self.indexes = np.arange(len(self.id_list))
        self.labelled_num = labelled_num

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        img = np.empty((self.batch_size, *self.dim, 1))
        ensemble_pred = np.zeros((self.batch_size, *self.dim, 5))
        flag = np.zeros((self.batch_size, *self.dim))

        pz_gt = np.zeros((self.batch_size, *self.dim))
        cz_gt = np.zeros((self.batch_size, *self.dim))
        us_gt = np.zeros((self.batch_size, *self.dim))
        afs_gt = np.zeros((self.batch_size, *self.dim))
        bg_gt = np.zeros((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img[i, :, :, :, :] = np.load(self.data_path + '/imgs/' + str(ID) + NPY)
            gt = np.load(self.data_path + '/gt/' + str(ID) + NPY)
            ensemble_pred[i] = np.load(self.ensemble_path + '/ens_gt/' + str(ID) + NPY)
            flag[i] = np.load(self.ensemble_path + '/flag/' + str(ID) + NPY).astype('int64')

            pz_gt[i] = gt[:, :, :, 0]
            cz_gt[i] = gt[:, :, :, 1]
            us_gt[i] = gt[:, :, :, 2]
            afs_gt[i] = gt[:, :, :, 3]
            bg_gt[i] = gt[:, :, :, 4]

        x_t = [img, ensemble_pred, flag]
        y_t = [pz_gt, cz_gt, us_gt, afs_gt, bg_gt]
        del gt, img, ensemble_pred, flag

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
        [img, ensemble_pred, flag], [pz_gt, cz_gt, us_gt, afs_gt, bg_gt] = self.__data_generation(list_IDs_temp)

        return [img, ensemble_pred, flag], [pz_gt, cz_gt, us_gt, afs_gt, bg_gt]
