import numpy as np
import random
from kits.utils import makedir

root_path = '/cache/suhita/data/prostate/'
fold_num = 2
import os

# training
perc = 1.0
labelled_num = 58
# data = '/cache/suhita/data/prostate/fold_' + str(fold_num)+'/'
data = '/cache/suhita/prostate/fold_' + str(fold_num) + '_supervised/'
print(len(os.listdir(os.path.join(data, 'train', 'imgs'))))
print(len(os.listdir(os.path.join(data, 'val', 'imgs'))))

ul_imgs_path = root_path + 'good_prediction_arr/'

num_labeled_train = int(perc * labelled_num)
labelled_num_considrd = [str(i) for i in np.arange(labelled_num)]
np.random.seed(1234)
np.random.shuffle(labelled_num_considrd)
labelled_num_considrd = labelled_num_considrd[0:num_labeled_train]

counter = 0
# training
makedir(root_path + 'fold_' + str(fold_num) + '_P' + str(perc) + '/train/imgs/')
makedir(root_path + 'fold_' + str(fold_num) + '_P' + str(perc) + '/train/gt/')
makedir(root_path + 'fold_' + str(fold_num) + '_P' + str(perc) + '/val/imgs/')
makedir(root_path + 'fold_' + str(fold_num) + '_P' + str(perc) + '/val/gt/')

for i in labelled_num_considrd:
    np.save(root_path + 'fold_' + str(fold_num) + '_P' + str(perc) + '/train/imgs/' + str(counter),
            np.load(os.path.join(data, 'train', 'imgs', i + '.npy')))
    np.save(root_path + 'fold_' + str(fold_num) + '_P' + str(perc) + '/train/gt/' + str(counter),
            np.load(os.path.join(data, 'train', 'gt', i + '.npy')))
    counter = counter + 1
    print(i, counter)

for i in np.arange(len(os.listdir(ul_imgs_path + '/imgs/'))):
    np.save(root_path + 'fold_' + str(fold_num) + '_P' + str(perc) + '/train/imgs/' + str(counter),
            np.load(os.path.join(ul_imgs_path, 'imgs', str(i) + '.npy')))
    np.save(root_path + 'fold_' + str(fold_num) + '_P' + str(perc) + '/train/gt/' + str(counter),
            np.load(os.path.join(ul_imgs_path, 'gt', str(i) + '.npy')))
    counter = counter + 1
    print(i, counter)

# counter=0
for i in np.arange(0, 20):
    np.save(root_path + 'fold_' + str(fold_num) + '_P' + str(perc) + '/val/imgs/' + str(i),
            np.load(os.path.join(data, 'val', 'imgs', str(i) + '.npy')))
    np.save(root_path + 'fold_' + str(fold_num) + '_P' + str(perc) + '/val/gt/' + str(i),
            np.load(os.path.join(data, 'val', 'gt', str(i) + '.npy')))
    # counter = counter + 1
    print(i)
