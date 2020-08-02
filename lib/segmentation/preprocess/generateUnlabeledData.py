import numpy as np

fold_num = 2
perc = 1.0
unlabeled_imgs = np.load('/cache/suhita/data/prostate/npy_img_unlabeled.npy')
unlabeled_imgs_gt = np.load('/data/suhita/prostate/supervised_F' + str(fold_num) + '_P' + str(perc) + '.npy')

good_count = unlabeled_imgs.shape[0] - 8

bad_prediction_arr = np.empty((8, 32, 168, 168, 1))
bad_prediction_arr_gt = np.empty((8, 32, 168, 168, 5))
good_prediction_arr = np.empty((good_count, 32, 168, 168, 1))
good_prediction_arr_gt = np.empty((good_count, 32, 168, 168, 5))

questionable = [47, 109, 203, 215]
bad = [99, 100, 101, 103]
all = np.arange(unlabeled_imgs.shape[0])

conc = questionable + bad
# conc =  bad
good_imgs_list = set(all) - set(conc)
counter = 0
for i in conc:
    bad_prediction_arr[counter] = unlabeled_imgs[i]
    # bad_prediction_arr_gt[counter] = np.transpose(unlabeled_imgs_gt[i], axes=(1, 2, 3, 0))
    bad_prediction_arr_gt[counter] = unlabeled_imgs_gt[i]
    print('bad images', i)
    counter += 1

np.save('/cache/suhita/bad_prediction_arr', bad_prediction_arr)
# np.save('/cache/suhita/bad_prediction_arr_gt', bad_prediction_arr_gt.astype('int8'))
import os, shutil


def makedir(dirpath):
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    try:
        os.makedirs(dirpath)
    except OSError:
        # [Errno 17] File exists
        pass


makedir('/cache/suhita/data/prostate/good_prediction_arr/imgs/')
makedir('/cache/suhita/data/prostate/good_prediction_arr/gt/')
counter = 0
for i in good_imgs_list:
    # good_prediction_arr[counter] = unlabeled_imgs[i]
    np.save('/cache/suhita/data/prostate/good_prediction_arr/imgs/' + str(counter) + '.npy', unlabeled_imgs[i])
    np.save('/cache/suhita/data/prostate/good_prediction_arr/gt/' + str(counter) + '.npy', unlabeled_imgs_gt[i])
    #good_prediction_arr_gt[counter] = unlabeled_imgs_gt[i]
    counter += 1
    print(i)

print('done')

#np.save('/cache/suhita/data/prostate/good_prediction_arr_gt', good_prediction_arr_gt.astype('int8'))
