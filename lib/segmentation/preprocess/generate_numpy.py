import numpy as np
root_path = '/cache/suhita/data/'

# training
train_num = 40

timgs = np.load(root_path + 'trainArray_imgs_fold1.npy')
print(timgs.shape)
tgt = np.load(root_path + 'trainArray_GT_fold1.npy')
tgt = tgt.astype('int8')

imgs = np.load(root_path + 'good_prediction_arr.npy')
gt = np.load(root_path + 'good_prediction_arr_gt.npy')
gt = gt.astype('int8')

stop = imgs.shape[0] + train_num
print(stop)

for i in np.arange(train_num):
    np.save(root_path + 'fold1_40/train/imgs/' + str(i), timgs[i])
    np.save(root_path + 'fold1_40/train/gt/' + str(i), tgt[i])
    print(i)

for i in np.arange(start=train_num, stop=stop):
    np.save(root_path + 'fold1_40/train/imgs/' + str(i), imgs[i - train_num])
    np.save(root_path + 'fold1_40/train/gt/' + str(i), gt[i - train_num])
    print(i)

# validation


vimgs = np.load(root_path + 'valArray_imgs_fold1.npy')
print(vimgs.shape)
vgt = np.load(root_path + 'valArray_GT_fold1.npy').astype('int8')
# vgt = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_GT.npy').astype('int8')

for i in np.arange(vimgs.shape[0]):
    np.save(root_path + 'fold1_40/val/imgs/' + str(i), vimgs[i, :, :, :, :])
    np.save(root_path + 'fold1_40/val/gt/' + str(i), vgt[i, :, :, :, :])
    print(i)

'''
#test
vimgs = np.load('/cache/suhita/data/final_test_array_imgs.npy')
print(vimgs.shape)
vgt = np.load('/cache/suhita/data/final_test_array_GT.npy').astype('int8')
# vgt = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_GT.npy').astype('int8')

for i in np.arange(vimgs.shape[0]):
    np.save(root_path + 'fold1_58/val/imgs/' + str(i), vimgs[i, :, :, :, :])
    np.save(root_path + 'fold1_58/val/gt/' + str(i), vgt[i, :, :, :, :])
    print(i)
'''
