import os

import numpy as np

from lib.segmentation.old.model_TemporalEns_MC import weighted_model
from lib.segmentation.utils import get_complete_array

MODEL_NAME = '/home/suhita/zonals/data/model.h5'
VAL_IMGS_PATH = '/home/suhita/zonals/data/test_anneke/imgs/'
VAL_GT_PATH = '/home/suhita/zonals/data/test_anneke/gt/'

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


w = weighted_model()
model = w.build_model(num_class=5, use_dice_cl=True, learning_rate=2.5e-5, gpu_id=None,
                      nb_gpus=None, trained_model=MODEL_NAME)
print('load_weights')
val_x_arr = get_complete_array(VAL_IMGS_PATH)
val_y_arr = get_complete_array(VAL_GT_PATH, dtype='int8')

val_supervised_flag = np.ones((val_x_arr.shape[0], 32, 168, 168), dtype='int8')

pz = val_y_arr[:, :, :, :, 0]
cz = val_y_arr[:, :, :, :, 1]
us = val_y_arr[:, :, :, :, 2]
afs = val_y_arr[:, :, :, :, 3]
bg = val_y_arr[:, :, :, :, 4]

y_val = [pz, cz, us, afs, bg]
x_val = [val_x_arr, val_y_arr, val_supervised_flag]

ENS_NO = 20
IMG_NUM = val_x_arr.shape[0]
out = np.empty((ENS_NO, IMG_NUM, 32, 168, 168, 5))
for i in np.arange(ENS_NO):
    arr = model.predict(x_val, batch_size=1, verbose=1)
    out[i, :, :, :, :, 0] = arr[0]
    out[i, :, :, :, :, 1] = arr[1]
    out[i, :, :, :, :, 2] = arr[2]
    out[i, :, :, :, :, 3] = arr[3]
    out[i, :, :, :, :, 4] = arr[4]
    # print(np.unique(out[i, :, :, :, :, 3]))
    print(i)

mean = np.empty((IMG_NUM, 32, 168, 168, 5))
var = np.empty((IMG_NUM, 32, 168, 168, 5))

for i in np.arange(ENS_NO):
    mean = mean + out[i]

mean = mean / ENS_NO

for i in np.arange(ENS_NO):
    var = var + np.square(out[i] - mean)

var = var / ENS_NO

np.save('mvariance_test_' + str(ENS_NO), var)
np.save('mmean_test_' + str(ENS_NO), mean)
