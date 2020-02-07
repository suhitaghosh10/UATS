import keras
import numpy as np
import SimpleITK as sitk

import hippocampus.AugmentationGenerator as aug
import os


def get_multi_class_arr(arr, n_classes = 3):
    size = arr.shape
    out_arr = np.zeros([size[0], size[1], size[2], n_classes])

    for i in range(n_classes):
        arr_temp = arr.copy()
        out_arr[:,:,:,i] =np.where(arr_temp == i, 1, 0)
        del arr_temp
    return out_arr

class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_path, id_list, batch_size=2, dim=(32, 168, 168), n_channels=1,
                 n_classes=1, shuffle=True, rotation=True, augmentation = False, GT_path = ''):
        'Initialization'
        self.dim = dim
        self.data_path = data_path
        self.GT_path = GT_path
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
        Y = np.empty((self.batch_size, *self.dim,self.n_classes), dtype=np.uint8)


        # Generate data
        for i, ID in enumerate(list_IDs_temp):



            orig_gt = np.load(os.path.join(self.GT_path, ID))

            if self.augmentation:
                aug_type = np.random.randint(0, 5)



                X[i], aug_gt = aug.get_single_image_augmentation(aug_type,
                                                                             np.load(os.path.join(self.data_path, ID)),
                                                                             orig_gt, img_no=ID)

                aug_gt = get_multi_class_arr(aug_gt[:,:,:,0], 3)
                Y[i] = aug_gt

            else:
                X[i, :, :, :, 0] = np.load(os.path.join(self.data_path, ID))
                Y[i] = get_multi_class_arr(orig_gt,3 )


        return X, [Y[:,:,:,:,0], Y[:,:,:,:,1], Y[:,:,:,:,2]]

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


