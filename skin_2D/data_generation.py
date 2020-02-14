import keras
import numpy as np
import SimpleITK as sitk
from keras.preprocessing.image import ImageDataGenerator

import os


def remove_tumor_segmentation(arr):
    arr[arr > 1] = 1
    return arr

def augment_image(img, augmentation_type, datagen):

    if augmentation_type==0: #rotation
        delta = np.random.uniform(0, 180)
        out = datagen.apply_transform(x=img,transform_parameters={'theta':delta})
    if augmentation_type==1: # translation
        [delta_x, delta_y] = [np.random.uniform(-15, 15), np.random.uniform(-10, 10)]  # in mm
        out = datagen.apply_transform(x=img, transform_parameters={'tx':delta_x, 'ty':delta_y})


    if augmentation_type==2: # scale
        scale_factor = np.random.uniform(1.0, 1.2)
        out = datagen.apply_transform(x=img, transform_parameters={'sx':scale_factor, 'sy':scale_factor})

    if augmentation_type == 3:
        out = img

    if augmentation_type== 4:
        out = datagen.apply_transform(x=img,transform_parameters={'flip_horizontal':True})

    if augmentation_type== 5:
        out = datagen.apply_transform(x=img,transform_parameters={'flip_vertical':True})

    return out

class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_path, id_list, batch_size=2, dim=(32, 168, 168), n_channels=1,
                 n_classes=1, shuffle=True, rotation=True, augmentation = False):
        'Initialization'
        self.dim = dim
        self.data_path = data_path
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.rotation = rotation
        self.id_list = id_list
        self.indexes = np.arange(len(self.id_list))
        self.augmentation = augmentation
        self.datagen_img = ImageDataGenerator(fill_mode='constant', cval=0, interpolation_order=1)
        self.datagen_GT = ImageDataGenerator(fill_mode='constant', cval=0, interpolation_order=0)


    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # 32,168,168
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype = float)
        Y = np.empty((self.batch_size, *self.dim,2), dtype=np.uint8)


        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            #Y[i, :, :, :, 0] = remove_tumor_segmentation(np.load(os.path.join(self.data_path, segm_file_name)))#[10:-10, 10:-10, 2:-2]

           # sitk.WriteImage(sitk.GetImageFromArray(X[i, :, :, :, 0]), 'img.nrrd')
           # sitk.WriteImage(sitk.GetImageFromArray(Y[i, :, :, :, 0]), 'segm.nrrd')
            x = np.load(os.path.join(self.data_path, 'imgs', ID))
            y = np.load(os.path.join(self.data_path, 'GT', ID[:-4] + '_segmentation.npy'))
            x = x / 255
            y = y / 255


            if self.augmentation:

                aug_type = np.random.randint(0, 6)

                X[i, :, :, :] = augment_image(x,aug_type, self.datagen_img)
                augm_GT = augment_image(y,aug_type, self.datagen_GT)[:,:,0]
                Y[i, :, :, 1] = augm_GT
                augm_GT_bg = np.where(augm_GT==0,1,0)
                Y[i, :, :, 0] = augm_GT_bg


            else:
                X[i] = x
                GT =  y[:,:,0]
                Y[i, :, :, 1] = GT
                GT_bg = np.where(GT == 0, 1, 0)
                Y[i, :, :, 0] = GT_bg


        return X, [Y[:,:,:,0],Y[:,:,:,1]]

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
