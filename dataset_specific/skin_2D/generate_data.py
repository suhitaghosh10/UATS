import os
from shutil import copyfile

import numpy as np

from dataset_specific.kits import utils

root_path = '/cache/suhita/skin/'
perc = 0.1
# training


fold_num = 3
labelled_path = root_path + 'preprocessed/labelled/train'
labelled_files_lst = np.load(root_path + 'Folds/train_fold' + str(fold_num) + '.npy')
labelled_train_num = len(labelled_files_lst)

un_labelled_path = '/data/suhita/skin/ul/UL_' + str(perc) + '/'
un_labelled_files_lst = os.listdir(un_labelled_path + '/imgs/')

print(labelled_files_lst[0:10])

# np.random.seed(5)
np.random.seed(1234)
np.random.shuffle(labelled_files_lst)

labelled_num_considrd = labelled_files_lst[:int(labelled_train_num * perc)]
remaining_labelled = set(labelled_files_lst).difference(set(labelled_num_considrd))

counter = 0

data_path = '/cache/suhita/data/skin/softmax/fold_' + str(fold_num) + '_P' + str(perc)
utils.makedir(data_path)
utils.makedir(os.path.join(data_path, 'imgs'))
utils.makedir(os.path.join(data_path, 'GT'))
for i in labelled_num_considrd:
    # name = labelled_files_lst[i]
    print(i, counter)
    np.save(os.path.join(data_path, 'imgs', str(counter) + '.npy'),
            np.load(os.path.join(labelled_path, 'imgs', i)) / 255
            )
    GT_lesion = np.load(os.path.join(labelled_path, 'GT', i.replace('.npy', '_segmentation.npy'))) / 255
    GT_bg = np.where(GT_lesion == 0, np.ones_like(GT_lesion), np.zeros_like(GT_lesion))

    np.save(os.path.join(data_path, 'GT', str(counter) + '.npy'),
            np.concatenate((GT_bg, GT_lesion), -1))
    counter = counter + 1

print('remaining labelled')
for i in remaining_labelled:
    # name = labelled_files_lst[i]
    print(i, counter)
    np.save(os.path.join(data_path, 'imgs', str(counter) + '.npy'),
            np.load(os.path.join(labelled_path, 'imgs', i)) / 255)
    GT_lesion = np.load(os.path.join(labelled_path, 'GT', i.replace('.npy', '_segmentation.npy'))) / 255
    GT_bg = np.where(GT_lesion == 0, np.ones_like(GT_lesion), np.zeros_like(GT_lesion))

    np.save(os.path.join(data_path, 'GT', str(counter) + '.npy'),
            np.concatenate((GT_bg, GT_lesion), -1))
    counter = counter + 1

print('unlabelled start...')

for i in un_labelled_files_lst:
    print(i, counter)
    copyfile(os.path.join(un_labelled_path, 'imgs', str(i)),
             os.path.join(data_path, 'imgs', str(counter) + '.npy'))
    copyfile(os.path.join(un_labelled_path, 'GT', str(i)),
             os.path.join(data_path, 'GT', str(counter) + '.npy'))

    counter = counter + 1
