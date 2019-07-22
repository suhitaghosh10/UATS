import matplotlib.pyplot as plt
import numpy as np

from zonal_utils.utils import get_val_id_list

TRAIN_IMGS_PATH = 'D:/Thesis/dataset_fold1/imgs/'
TRAIN_GT_PATH = 'D:/Thesis/dataset_fold1/gt/'
v_list = get_val_id_list(1)


def get_array_from_list(folder_path, imgs_no=[], dtype=None):
    arr = np.load(folder_path + '/0.npy')
    size = len(imgs_no)
    if dtype is None:
        total_arr = np.zeros((size, *arr.shape))
    else:
        total_arr = np.zeros((size, *arr.shape), dtype=dtype)
    start = 0
    for idx in imgs_no:
        total_arr[start] = np.load(folder_path + '/' + str(idx) + '.npy')
        start = start + 1

    return total_arr


val_x_arr = get_array_from_list(TRAIN_IMGS_PATH, v_list)
val_y_arr = get_array_from_list(TRAIN_GT_PATH, v_list, dtype='int8')
img_no = 5
imgs = val_x_arr[img_no]
gt = val_y_arr[img_no]

print(gt.shape)
slice = 16
z = 0
plt.imshow(imgs[slice, :, :, 0], cmap='gray')
plt.imshow(gt[16, :, :, z], alpha=0.6, cmap='coolwarm')
plt.show()
