import numpy as np
import random
import os
from shutil import copyfile
from kits import utils

root_path = '/cache/suhita/skin/'
perc = 1.0
# training


fold_num = 1
labelled_path = root_path + 'preprocessed/labelled/train'
labelled_files_lst = np.load(root_path + 'Folds/train_fold' + str(fold_num) + '.npy')
labelled_train_num = len(labelled_files_lst)

un_labelled_path = '/data/suhita/skin/UL_' + str(perc) + '/'
un_labelled_files_lst = os.listdir(un_labelled_path + '/imgs/')

print(labelled_files_lst[0:10])

# np.random.seed(5)
np.random.seed(1234)
np.random.shuffle(labelled_files_lst)

labelled_num_considrd = labelled_files_lst[:int(labelled_train_num * perc)]
remaining_labelled = set(labelled_files_lst).difference(set(labelled_num_considrd))

counter = 0

data_path = '/cache/suhita/data/skin/fold_' + str(fold_num) + '_P' + str(perc)
utils.makedir(data_path)
utils.makedir(os.path.join(data_path, 'imgs'))
utils.makedir(os.path.join(data_path, 'GT'))
for i in labelled_num_considrd:
    # name = labelled_files_lst[i]
    print(i, counter)
    np.save(os.path.join(data_path, 'imgs', str(counter) + '.npy'),
            np.load(os.path.join(labelled_path, 'imgs', i)) / 255
            )
    np.save(os.path.join(data_path, 'GT', str(counter) + '.npy'),
            np.load(os.path.join(labelled_path, 'GT', i.replace('.npy', '_segmentation.npy'))) / 255)
    counter = counter + 1

print('remaining labelled')
for i in remaining_labelled:
    # name = labelled_files_lst[i]
    print(i, counter)
    np.save(os.path.join(data_path, 'imgs', str(counter) + '.npy'),
            np.load(os.path.join(labelled_path, 'imgs', i)) / 255)
    np.save(os.path.join(data_path, 'GT', str(counter) + '.npy'),
            np.load(os.path.join(labelled_path, 'GT', i.replace('.npy', '_segmentation.npy'))) / 255
            )
    counter = counter + 1

print('unlabelled start...')

for i in np.arange(len(un_labelled_files_lst)):
    print(i, counter)
    np.save(os.path.join(data_path, 'imgs', str(counter) + '.npy'),
            np.load(os.path.join(un_labelled_path, 'imgs', str(i) + '.npy'))
            )
    copyfile(os.path.join(un_labelled_path, 'GT', str(i) + '.npy'),
             os.path.join(data_path, 'GT', str(counter) + '.npy'))

    counter = counter + 1
