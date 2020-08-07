import os

import numpy as np
from keras.callbacks import Callback

from utility.constants import ENS_GT, NPY, IMGS, ALPHA
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
        self.batch_size = batch_size
        self.nr_dim = len(dim)

        makedir(self.temp_path)
        makedir(os.path.join(self.temp_path, ENS_GT))
        makedir(os.path.join(save_path, 'tb', dataset_name))
        makedir(os.path.join(save_path, 'csv', dataset_name))
        makedir(os.path.join(save_path, 'model', dataset_name))

        ens_gt_shape = (dim[0], dim[1], nr_class) if self.nr_dim == 2 else (dim[0], dim[1], dim[2], nr_class)
        ens = np.zeros(ens_gt_shape)
        for patient in np.arange(num_train_data):
            np.save(os.path.join(self.temp_path, ENS_GT, str(patient) + NPY), ens)

    def on_epoch_end(self, epoch, logs={}):

        if epoch > -1:

            # patients_per_batch = 59
            num_batches = self.num_un_labeled_train // self.patients_per_batch
            remainder = self.num_un_labeled_train % self.patients_per_batch
            patients_in_last_batch = self.patients_per_batch if remainder == 0 else (
                    self.patients_per_batch + remainder)

            for b_no in range(num_batches):
                actual_batch_size = self.patients_per_batch if (
                        b_no < num_batches - 1) else patients_in_last_batch

                start = (b_no * self.patients_per_batch) + self.num_labeled_train
                end = (start + actual_batch_size)
                imgs = get_array(os.path.join(self.data_path, IMGS), start, end)
                ensemble_prediction = get_array(os.path.join(self.temp_path, ENS_GT), start, end, dtype='float32')

                inp = [imgs, ensemble_prediction]
                del imgs

                cur_pred = np.zeros((actual_batch_size, self.dim[0], self.dim[1], self.nr_class)) if self.nr_dim == 2 \
                    else np.zeros((actual_batch_size, self.dim[0], self.dim[1], self.dim[2], self.nr_class))

                model_out = self.model.predict(inp, batch_size=self.batch_size, verbose=1)
                del inp

                for idx in range(self.nr_class):
                    cur_pred[:, :, :, :, idx] = model_out[idx]

                del model_out

                # Z = αZ + (1 - α)z
                ensemble_prediction = ALPHA * ensemble_prediction + (1 - ALPHA) * cur_pred
                save_array(os.path.join(self.temp_path, ENS_GT), ensemble_prediction, start, end)
                del ensemble_prediction

            if 'cur_pred' in locals(): del cur_pred
