
import numpy as np

timgs = np.load('D:/Thesis/numpy/train/good_prediction_arr_gt.npy').astype('int8')
print(timgs.shape)
np.save('D:/Thesis/numpy/train/good_prediction_arr_gt_c.npy', timgs)