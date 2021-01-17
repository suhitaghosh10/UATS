import os

import numpy as np

import dataset_specific.prostate.model.uats_scaled as uats
import dataset_specific.prostate.model.baseline as sup
from old.utils.preprocess_images import get_complete_array
import nrrd
import itk
import math

#MODEL_NAME = '/data/suhita/experiments/model/uats/prostate/uats_softmax_F2_Perct_Labelled_1.0.h5'
MODEL_NAME = '/data/suhita/experiments/model/supervised/prostate/supervised_F2_P0.5.h5'
VAL_IMGS_PATH = '/cache/suhita/data/prostate/final_test_array_imgs.npy'
VAL_GT_PATH = '/cache/suhita/data/prostate/final_test_array_GT.npy'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

w = sup.weighted_model()
#w= uats.weighted_model()
model = w.build_model(trained_model=MODEL_NAME)
print('load_weights')
val_x_arr = np.load(VAL_IMGS_PATH)
val_y_arr = np.load(VAL_GT_PATH)

val_supervised_flag = np.ones((val_x_arr.shape[0], 32, 168, 168), dtype='int8')

pz = val_y_arr[:, :, :, :, 0]
cz = val_y_arr[:, :, :, :, 1]
us = val_y_arr[:, :, :, :, 2]
afs = val_y_arr[:, :, :, :, 3]
bg = val_y_arr[:, :, :, :, 4]

y_val = [pz, cz, us, afs, bg]
#x_val = [val_x_arr, val_y_arr, val_supervised_flag]
x_val = [val_x_arr]

ENS_NO = 25
IMG_NUM = val_x_arr.shape[0]
# out = np.empty((ENS_NO, IMG_NUM, 32, 168, 168, 5))
# for i in np.arange(ENS_NO):
#      arr = model.predict(x_val, batch_size=1, verbose=1)
#      out[i, :, :, :, :, 0] = arr[0]
#      out[i, :, :, :, :, 1] = arr[1]
#      out[i, :, :, :, :, 2] = arr[2]
#      out[i, :, :, :, :, 3] = arr[3]
#      out[i, :, :, :, :, 4] = arr[4]
     # print(np.unique(out[i, :, :, :, :, 3]))
     #print(i)

mc_pred = np.zeros((IMG_NUM, 32, 168, 168, 5))
for i in np.arange(ENS_NO):
    model_out = model.predict(x_val, batch_size=2, verbose=1)
    for cls in range(5):
        mc_pred[:, :, :, :, cls] = np.add(model_out[cls], mc_pred[:, :, :, :, cls])

#mean = np.empty((IMG_NUM, 32, 168, 168, 5))
entropy = np.empty((IMG_NUM, 32, 168, 168))
#var = np.empty((IMG_NUM, 32, 168, 168, 5))

# for i in np.arange(ENS_NO):
#     mean = mean + out[i]

# mean = mean / ENS_NO

 # for i in np.arange(ENS_NO):
 #     var = var + np.square(out[i] - mean)

for z in np.arange(5):
    if z == 0:
        entropy = (mc_pred[:, :, :, :, z] / ENS_NO) * ((np.log(mc_pred[:, :, :, :, z] / ENS_NO)) /  np.log(5))
    else:
        entropy = entropy + (mc_pred[:, :, :, :, z] / ENS_NO) * ((np.log(mc_pred[:, :, :, :, z] / ENS_NO)) /  np.log(5))
entropy = -entropy

#var = var / ENS_NO
#var = var * 100

for p in range(IMG_NUM):
    for z in range(4):
        # image = itk.GetImageFromArray(mean[p,:,:,:,z])
        # itk.imwrite(image, '/data/suhita/experiments/mc_results/uats/M_test_zone'+str(z)+'_img'+str(p)+'.nrrd')
        image = itk.GetImageFromArray(entropy[p, :, :, :])
        itk.imwrite(image,
                     '/data/suhita/experiments/mc_results/supervised/entropy/V_test_img'+ str(p) + '.nrrd')

# for p in range(IMG_NUM):
#     image = itk.GetImageFromArray(val_x_arr[p,:,:,:,0])
#     itk.imwrite(image, '/data/suhita/experiments/mc_results/IMG_' + str(p) + '.nrrd')
#
#     for z in range(4):
#         image = itk.GetImageFromArray(val_y_arr[p,:,:,:,z])
#         itk.imwrite(image, '/data/suhita/experiments/mc_results/GT_'+str(p)+'_Z'+str(z)+'.nrrd')