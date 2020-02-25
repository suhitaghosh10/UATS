import numpy as np

unlabeled_imgs = np.load('/cache/suhita/data/prostate/npy_img_unlabeled.npy')
unlabeled_imgs_gt = np.load('/data/suhita/prostate/supervised_F2_P_0.5.npy')

good_count = unlabeled_imgs.shape[0] - 8

bad_prediction_arr = np.empty((8, 32, 168, 168, 1))
bad_prediction_arr_gt = np.empty((8, 32, 168, 168, 5))
good_prediction_arr = np.empty((good_count, 32, 168, 168, 1))
good_prediction_arr_gt = np.empty((good_count, 32, 168, 168, 5))

questionable = [47, 109, 203, 215]
bad = [99, 100, 101, 103]
all = np.arange(unlabeled_imgs.shape[0])

conc = questionable + bad
good_imgs_list = set(all) - set(conc)
counter = 0
for i in conc:
    bad_prediction_arr[counter] = unlabeled_imgs[i]
    # bad_prediction_arr_gt[counter] = np.transpose(unlabeled_imgs_gt[i], axes=(1, 2, 3, 0))
    bad_prediction_arr_gt[counter] = unlabeled_imgs_gt[i]
    print('bad images', i)
    counter += 1

# np.save('/home/suhita/zonals/data/training/bad/bad_prediction_arr', bad_prediction_arr)
#np.save('/home/suhita/zonals/data/training/bad/bad_prediction_arr_gt', bad_prediction_arr_gt.astype('int8'))

counter = 0
for i in good_imgs_list:
    good_prediction_arr[counter] = unlabeled_imgs[i]
    good_prediction_arr_gt[counter] = unlabeled_imgs_gt[i]
    counter += 1
    print(i)

np.save('/cache/suhita/data/prostate/good_prediction_arr', good_prediction_arr)
np.save('/cache/suhita/data/prostate/good_prediction_arr_gt', good_prediction_arr_gt.astype('int8'))
