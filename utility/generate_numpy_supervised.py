import numpy as np

from utility.utils import makedir

root_path = '/cache/suhita/data/prostate/'

# training
train_num = 58
fold = 1
timgs = np.load(root_path + 'trainArray_imgs_fold1.npy')
print(timgs.shape)
tgt = np.load(root_path + 'trainArray_GT_fold1.npy')
tgt = tgt.astype('int8')
counter = 0

makedir(root_path + 'fold_' + str(fold) + '/train/imgs/')
makedir(root_path + 'fold_' + str(fold) + '/train/gt/')
makedir(root_path + 'fold_' + str(fold) + '/val/imgs/')
makedir(root_path + 'fold_' + str(fold) + '/val/gt/')

for i in np.arange(58):
    np.save(root_path + 'fold_' + str(fold) + '/train/imgs/' + str(counter), timgs[i])
    np.save(root_path + 'fold_' + str(fold) + '/train/gt/' + str(counter), tgt[i])
    counter = counter + 1
    print(i, counter)


# validation
vimgs = np.load(root_path + 'valArray_imgs_fold1.npy')
print(vimgs.shape)
vgt = np.load(root_path + 'valArray_GT_fold1.npy').astype('int8')

for i in np.arange(vimgs.shape[0]):
    np.save(root_path + 'fold_' + str(fold) + '/val/imgs/' + str(i), vimgs[i, :, :, :, :])
    np.save(root_path + 'fold_' + str(fold) + '/val/gt/' + str(i), vgt[i, :, :, :, :])
    print(i)


#test
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
