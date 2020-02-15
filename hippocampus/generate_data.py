import numpy as np
import random
import os
from shutil import copyfile
from kits import utils
from hippocampus.utils import get_multi_class_arr

root_path = '/cache/suhita/hippocampus/'
perc = 1.0
# training


fold_num = 2
labelled_imgs_path = '/cache/suhita/hippocampus/preprocessed/labelled/train/'
labelled_gt_path = '/cache/suhita/hippocampus/preprocessed/labelled-GT/train/'
labelled_files_lst = np.load(root_path + 'Folds/train_fold' + str(fold_num) + '.npy')
labelled_train_num = len(labelled_files_lst)
# hippocampus_096.nii.gz
un_labelled_path = '/data/suhita/hippocampus/UL_' + str(perc) + '/'
un_labelled_files_lst = os.listdir(un_labelled_path + '/imgs/')

print(labelled_files_lst[0:10])

# np.random.seed(5)
np.random.seed(5)
np.random.shuffle(labelled_files_lst)

labelled_num_considrd = labelled_files_lst[:int(labelled_train_num * perc)]
remaining_labelled = set(labelled_files_lst).difference(set(labelled_num_considrd))

counter = 0

data_path = '/cache/suhita/data/hippocampus/fold_' + str(fold_num) + '_P' + str(perc)

utils.makedir(data_path)
utils.makedir(os.path.join(data_path, 'imgs'))
utils.makedir(os.path.join(data_path, 'GT'))
for i in labelled_num_considrd:
    # name = labelled_files_lst[i]
    print(i, counter)
    np.save(os.path.join(data_path, 'imgs', str(counter) + '.npy'),
            np.expand_dims(np.load(os.path.join(labelled_imgs_path, i.replace('.nii.gz', '.npy'))), -1))
    np.save(os.path.join(data_path, 'GT', str(counter) + '.npy'),
            get_multi_class_arr(np.load(os.path.join(labelled_gt_path, i.replace('.nii.gz', '.npy'))), 3))
    counter = counter + 1

print('remaining labelled')
for i in remaining_labelled:
    # name = labelled_files_lst[i]
    print(i, counter)
    np.save(os.path.join(data_path, 'imgs', str(counter) + '.npy'),
            np.expand_dims(np.load(os.path.join(labelled_imgs_path, i.replace('.nii.gz', '.npy'))), -1))
    np.save(os.path.join(data_path, 'GT', str(counter) + '.npy'),
            get_multi_class_arr(np.load(os.path.join(labelled_gt_path, i.replace('.nii.gz', '.npy'))), 3))
    counter = counter + 1

print('unlabelled start...')

for i in np.arange(len(un_labelled_files_lst)):
    print(i, counter)
    np.save(os.path.join(data_path, 'imgs', str(counter) + '.npy'),
            np.load(os.path.join(un_labelled_path, 'imgs', str(i) + '.npy'))
            )
    np.save(os.path.join(data_path, 'GT', str(counter) + '.npy'),
            np.transpose(np.load(os.path.join(un_labelled_path, 'GT', str(i) + '.npy')), axes=(1, 2, 3, 0))
            )

    counter = counter + 1
