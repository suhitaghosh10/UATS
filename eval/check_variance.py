import os

import numpy as np

from lib.segmentation.model_TemporalEns import weighted_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
w = weighted_model()
model = w.build_model(num_class=5, use_dice_cl=True, learning_rate=4.5e-5)
print('load_weights')
model.load_weights('/home/suhita/zonals/data/model.h5')
unl_x = np.load('/home/suhita/zonals/data/training/bad_prediction_arr.npy')

val_unsupervised_target = np.zeros((8, 32, 168, 168, 5))
val_supervised_flag = np.ones((8, 32, 168, 168, 1))
val_unsupervised_weight = np.zeros((8, 32, 168, 168, 5))

x_val = [unl_x, val_unsupervised_target, val_supervised_flag, val_unsupervised_weight]

ENS_NO = 20
out = np.empty((ENS_NO, 8, 32, 168, 168, 5))
for i in np.arange(ENS_NO):
    arr = model.predict(x_val, batch_size=1, verbose=1)
    out[i, :, :, :, :, 0] = arr[0]
    out[i, :, :, :, :, 1] = arr[1]
    out[i, :, :, :, :, 2] = arr[2]
    out[i, :, :, :, :, 3] = arr[3]
    out[i, :, :, :, :, 4] = arr[4]

mean = np.empty((8, 32, 168, 168, 5))
var = np.empty((8, 32, 168, 168, 5))

for i in np.arange(ENS_NO):
    mean = mean + out[i]

mean = mean / ENS_NO

for i in np.arange(ENS_NO):
    var = var + np.square(out[i] - mean)

var = var / ENS_NO

np.save('variance_bas', var)
np.save('mean_bas', mean.astype('float32'))
