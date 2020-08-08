import os
from shutil import copyfile

import numpy as np
from keras.callbacks import Callback

from utility.config import UPDATE_EPOCH_NO
from utility.constants import ENS_GT, FLAG, GT, NPY, IMGS
from utility.utils import makedir, get_array, save_array


class TemporalCallback(Callback):

    def __init__(self, dim, data_path, temp_path, save_path, num_train_data, num_labeled_train,
                 patients_per_batch, nr_class, batch_size, dataset_name):

        self.data_path = data_path
        self.temp_path = temp_path
        self.save_path = save_path
        self.num_train_data = num_train_data  # total training data = Labelled + UnLabelled
        self.num_labeled_train = num_labeled_train  # labelled training data
        self.num_un_labeled_train = num_train_data - num_labeled_train  # unlabelled data
        self.patients_per_batch = patients_per_batch
        self.nr_class = nr_class
        self.dim = dim
        self.nr_dim = len(dim)
        self.batch_size = batch_size

        flag_1 = np.ones(shape=dim, dtype='int64')

        makedir(self.temp_path)
        makedir(os.path.join(self.temp_path, ENS_GT))
        makedir(os.path.join(self.temp_path, FLAG))
        makedir(os.path.join(save_path, 'tb', dataset_name))
        makedir(os.path.join(save_path, 'csv', dataset_name))
        makedir(os.path.join(save_path, 'model', dataset_name))

        for patient in np.arange(num_train_data):
            copyfile(os.path.join(data_path, GT, str(patient) + NPY),
                     os.path.join(self.temp_path, ENS_GT, str(patient) + NPY))

            np.save(os.path.join(self.temp_path, FLAG, str(patient) + NPY), flag_1)

    def on_epoch_end(self, epoch, logs={}):

        if epoch > 0 and epoch % UPDATE_EPOCH_NO == 0:

            # patients_per_batch = 59
            num_batches = self.num_un_labeled_train // self.patients_per_batch
            remainder = self.num_un_labeled_train % self.patients_per_batch

            if self.nr_dim == 2:
                remainder_pixels = remainder * self.dim[0] * self.dim[1]
                # calculate maximum number of confident pixels (background)
                confident_pixels_no_per_batch_bg = (self.pixel_perc_arr[-1] * self.patients_per_batch * self.dim[0] *
                                                    self.dim[1]) // 100
                total_pixels = self.dim[0] * self.dim[1]
            else:
                remainder_pixels = remainder * self.dim[0] * self.dim[1] * self.dim[2]
                # calculate maximum number of confident pixels (background)
                confident_pixels_no_per_batch_bg = (self.pixel_perc_arr[-1] * self.patients_per_batch * self.dim[0] *
                                                    self.dim[1] * self.dim[2]) // 100
                total_pixels = self.dim[0] * self.dim[1] * self.dim[2]

            if remainder_pixels < confident_pixels_no_per_batch_bg:
                patients_in_last_batch = self.patients_per_batch + remainder
            else:
                patients_in_last_batch = remainder
                num_batches = num_batches + 1

            for b_no in range(num_batches):
                actual_batch_size = self.patients_per_batch if (
                        b_no < num_batches - 1) else patients_in_last_batch

                start = (b_no * self.patients_per_batch) + self.num_labeled_train
                end = (start + actual_batch_size)
                imgs = get_array(os.path.join(self.data_path + IMGS), start, end)
                ensemble_prediction = get_array(os.path.join(self.temp_path + ENS_GT), start, end, dtype='float32')
                supervised_flag = get_array(os.path.join(self.temp_path + FLAG), start, end, dtype='int64')

                inp = [imgs, ensemble_prediction, supervised_flag]
                del imgs, supervised_flag

                cur_pred = np.zeros((actual_batch_size, self.dim[0], self.dim[1], self.nr_class)) if self.nr_dim == 2 \
                    else np.zeros((actual_batch_size, self.dim[0], self.dim[1], self.dim[2], self.nr_class))

                model_out = self.model.predict(inp, batch_size=self.batch_size, verbose=1)
                del inp

                for idx in range(self.nr_class):
                    cur_pred[:, :, :, :, idx] = model_out[idx]

                save_array(os.path.join(self.temp_path, ENS_GT), cur_pred, start, end)
                del ensemble_prediction, cur_pred, model_out

            if 'cur_pred' in locals(): del cur_pred
