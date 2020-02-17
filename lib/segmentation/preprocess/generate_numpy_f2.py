import numpy as np
import random
from kits.utils import makedir

root_path = '/cache/suhita/data/'
save_dir = '/cache/suhita/prostate/'
fold_num = 2

# training
perc = 1.0
labelled_num = 58
timgs = np.load(root_path + 'trainArray_imgs_fold1.npy')
print(timgs.shape)
tgt = np.load(root_path + 'trainArray_GT_fold1.npy')
tgt = tgt.astype('int8')

vimgs = np.load(root_path + 'valArray_imgs_fold1.npy')
print(vimgs.shape)
vgt = np.load(root_path + 'valArray_GT_fold1.npy').astype('int8')

imgs = np.load(root_path + 'good_prediction_arr.npy')
gt = np.load(root_path + 'good_prediction_arr_gt.npy')
gt = gt.astype('int8')

temp1 = set(np.arange(20))
temp2 = set(np.arange(40, 58))
labelled_num_considrd = temp1.union(temp2)

val = np.arange(20, 40)
counter = 0
# training
makedir(save_dir + 'fold_' + str(fold_num) + '_P' + str(perc) + '/train/imgs/')
makedir(save_dir + 'fold_' + str(fold_num) + '_P' + str(perc) + '/train/gt/')
makedir(save_dir + 'fold_' + str(fold_num) + '_P' + str(perc) + '/val/imgs/')
makedir(save_dir + 'fold_' + str(fold_num) + '_P' + str(perc) + '/val/gt/')

for i in labelled_num_considrd:
    np.save(save_dir + 'fold_' + str(fold_num) + '_P' + str(perc) + '/train/imgs/' + str(counter), timgs[i])
    np.save(save_dir + 'fold_' + str(fold_num) + '_P' + str(perc) + '/train/gt/' + str(counter), tgt[i])
    counter = counter + 1
    print(i, counter)

for i in np.arange(vimgs.shape[0]):
    np.save(save_dir + 'fold_' + str(fold_num) + '_P' + str(perc) + '/train/imgs/' + str(counter), vimgs[i, :, :, :, :])
    np.save(save_dir + 'fold_' + str(fold_num) + '_P' + str(perc) + '/train/gt/' + str(counter), vgt[i, :, :, :, :])
    counter = counter + 1
    print(i)

# val
counter = 0
for i in val:
    np.save(save_dir + 'fold_' + str(fold_num) + '_P' + str(perc) + '/val/imgs/' + str(counter), timgs[i])
    np.save(save_dir + 'fold_' + str(fold_num) + '_P' + str(perc) + '/val/gt/' + str(counter), tgt[i])
    counter = counter + 1
    print(i)

# validation


# vgt = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_GT.npy').astype('int8')


# test
'''
vimgs = np.load('/cache/suhita/data/final_test_array_imgs.npy')
print(vimgs.shape)
vgt = np.load('/cache/suhita/data/final_test_array_GT.npy').astype('int8')
# vgt = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_GT.npy').astype('int8')

for i in np.arange(vimgs.shape[0]):
    np.save(root_path + 'fold1_58/val/imgs/' + str(i), vimgs[i, :, :, :, :])
    np.save(root_path + 'fold1_58/val/gt/' + str(i), vgt[i, :, :, :, :])
    print(i)
    '''
