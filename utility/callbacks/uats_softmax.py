import os
from shutil import copyfile

import numpy as np
from keras.callbacks import Callback

from utility.constants import ENS_GT, FLAG, GT, NPY, IMGS, ALPHA
from utility.utils import makedir, shall_save, get_array, save_array


class TemporalCallback(Callback):

    def __init__(self, dim, data_path, temp_path, save_path, num_train_data, num_labeled_train,
                 patients_per_batch, pixel_perc_arr, val_metric_keys, nr_class, batch_size, dataset_name):

        self.data_path = data_path
        self.temp_path = temp_path
        self.save_path = save_path
        self.num_train_data = num_train_data  # total training data = Labelled + UnLabelled
        self.num_labeled_train = num_labeled_train  # labelled training data
        self.num_un_labeled_train = num_train_data - num_labeled_train  # unlabelled data
        self.patients_per_batch = patients_per_batch
        self.nr_class = nr_class
        self.pixel_perc_arr = pixel_perc_arr
        self.val_metric_keys = val_metric_keys
        self.save_flag = np.zeros((self.nr_class), dtype=np.bool)
        self.val_dice_coef = np.zeros((self.nr_class), dtype=np.float)
        self.confident_pixels_no_per_batch = np.zeros([self.nr_class], dtype=np.int)
        self.dim = dim
        self.batch_size = batch_size
        self.nr_dim = len(dim)

        flag_1 = np.ones(shape=dim, dtype='int64')
        flag_0 = np.zeros(shape=dim, dtype='int64')

        makedir(self.temp_path)
        makedir(os.path.join(self.temp_path, ENS_GT))
        makedir(os.path.join(self.temp_path, FLAG))
        makedir(os.path.join(save_path, 'tb', dataset_name))
        makedir(os.path.join(save_path, 'csv', dataset_name))
        makedir(os.path.join(save_path, 'model', dataset_name))

        for patient in np.arange(num_train_data):

            copyfile(os.path.join(data_path, GT, str(patient) + NPY),
                     os.path.join(self.temp_path, ENS_GT, str(patient) + NPY))

            if patient < num_labeled_train:
                np.save(os.path.join(self.temp_path, FLAG, str(patient) + NPY), flag_1)
            else:
                np.save(os.path.join(self.temp_path, FLAG, str(patient) + NPY), flag_0)
        del flag_1, flag_0

    def on_epoch_end(self, epoch, logs={}):

        for idx in range(self.nr_class):
            self.save_flag[idx], self.val_dice_coef[idx] = shall_save(logs[self.val_metric_keys[idx]],
                                                                      self.val_dice_coef[idx])

        if epoch > 0:

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

                for idx in range(self.nr_class):
                    self.confident_pixels_no_per_batch[idx] = (self.pixel_perc_arr[
                                                                   idx] * actual_batch_size * total_pixels) // 100

                start = (b_no * self.patients_per_batch) + self.num_labeled_train
                end = (start + actual_batch_size)
                imgs = get_array(os.path.join(self.data_path, IMGS), start, end)
                ensemble_prediction = get_array(os.path.join(self.temp_path, ENS_GT), start, end, dtype='float32')
                supervised_flag = get_array(os.path.join(self.temp_path, FLAG), start, end, dtype='int64')

                inp = [imgs, ensemble_prediction, supervised_flag]
                del imgs, supervised_flag

                cur_pred = np.zeros((actual_batch_size, self.dim[0], self.dim[1], self.nr_class)) if self.nr_dim == 2 \
                    else np.zeros((actual_batch_size, self.dim[0], self.dim[1], self.dim[2], self.nr_class))

                model_out = self.model.predict(inp, batch_size=self.batch_size, verbose=1)
                del inp

                for idx in range(self.nr_class):
                    cur_pred[:, :, :, :, idx] = model_out[idx] if self.save_flag[idx] else ensemble_prediction[:, :, :,
                                                                                           :, idx]

                del model_out

                # Z = αZ + (1 - α)z
                ensemble_prediction = ALPHA * ensemble_prediction + (1 - ALPHA) * cur_pred
                save_array(os.path.join(self.temp_path, ENS_GT), ensemble_prediction, start, end)
                del ensemble_prediction

                argmax_pred_ravel = np.ravel(np.argmax(cur_pred, axis=-1))
                max_pred_ravel = np.ravel(np.max(cur_pred, axis=-1))
                indices = None
                del cur_pred
                for zone in np.arange(self.nr_class):
                    final_max_ravel = np.where(argmax_pred_ravel == zone, np.zeros_like(max_pred_ravel),
                                               max_pred_ravel)
                    zone_indices = np.argpartition(final_max_ravel, -self.confident_pixels_no_per_batch[zone])[
                                   -self.confident_pixels_no_per_batch[zone]:]
                    indices = zone_indices if zone == 0 else np.unique(np.concatenate((zone_indices, indices)))

                mask = np.ones(max_pred_ravel.shape, dtype=bool)
                mask[indices] = False

                max_pred_ravel[mask] = 0
                max_pred_ravel = np.where(max_pred_ravel > 0, np.ones_like(max_pred_ravel) * 2,
                                          np.zeros_like(max_pred_ravel))

                flag = np.reshape(max_pred_ravel,
                                  (actual_batch_size, self.dim[0], self.dim[1])) if self.nr_dim == 2 else \
                    np.reshape(max_pred_ravel, (actual_batch_size, self.dim[0], self.dim[1], self.dim[2]))
                # flag = np.reshape(max_pred_ravel, (IMGS_PER_ENS_BATCH, 32, 168, 168))
                del max_pred_ravel, indices

                save_array(os.path.join(self.temp_path, FLAG), flag, start, end)

                del flag

            if 'cur_pred' in locals(): del cur_pred
