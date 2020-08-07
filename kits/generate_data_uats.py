import os
from shutil import copyfile

import numpy as np

from kits import utils

root_path = '/cache/suhita/data/kits/'
perc = 1.0
# training


fold_num = 1
labelled_path = '/data/suhita/temporal/kits/preprocessed_labeled_train'
labelled_fold_num = '/data/suhita/temporal/kits/Folds/train_fold' + str(fold_num) + '.npy'
un_labelled_path = '/data/suhita/temporal/kits/output/UL_' + str(perc) + '_PP/'
labelled_train_num = np.load(labelled_fold_num).shape[0]

labelled_num = np.load(labelled_fold_num).shape[0]
labelled_files_lst = np.load(labelled_fold_num)
un_labelled_files_lst = os.listdir(un_labelled_path)

train_fold = np.load('/data/suhita/temporal/kits/Folds/train_fold' + str(fold_num) + '.npy')
print(train_fold[0:10])
nr_samples = train_fold.shape[0]

# np.random.seed(5)
np.random.seed(5)
np.random.shuffle(train_fold)
print(train_fold[0:10])

labelled_num_considrd = train_fold[:int(nr_samples * perc)]
remaining_labelled = set(train_fold).difference(set(labelled_num_considrd))

counter = 0

data_path = '/cache/suhita/data/kits/fold_' + str(fold_num) + '_P' + str(perc)
utils.makedir(data_path)
for i in labelled_num_considrd:
    # name = labelled_files_lst[i]
    print(i, counter)
    fold_name = 'case_' + str(counter)
    utils.makedir(os.path.join(data_path, fold_name))
    copyfile(os.path.join(labelled_path, i, 'img_left.npy'), os.path.join(data_path, fold_name, 'img_left.npy'))
    copyfile(os.path.join(labelled_path, i, 'img_right.npy'),
             os.path.join(data_path, fold_name, 'img_right.npy'))
    copyfile(os.path.join(labelled_path, i, 'segm_left.npy'),
             os.path.join(data_path, fold_name, 'segm_left.npy'))
    copyfile(os.path.join(labelled_path, i, 'segm_right.npy'),
             os.path.join(data_path, fold_name, 'segm_right.npy'))
    counter = counter + 1

print('remianing labelled')
for i in remaining_labelled:
    # name = labelled_files_lst[i]
    print(i, counter)
    fold_name = 'case_' + str(counter)
    utils.makedir(os.path.join(data_path, fold_name))
    copyfile(os.path.join(labelled_path, i, 'img_left.npy'), os.path.join(data_path, fold_name, 'img_left.npy'))
    copyfile(os.path.join(labelled_path, i, 'img_right.npy'),
             os.path.join(data_path, fold_name, 'img_right.npy'))
    copyfile(os.path.join(labelled_path, i, 'segm_left.npy'),
             os.path.join(data_path, fold_name, 'segm_left.npy'))
    copyfile(os.path.join(labelled_path, i, 'segm_right.npy'),
             os.path.join(data_path, fold_name, 'segm_right.npy'))
    counter = counter + 1

print('unlabelled start...')

for i in np.arange(len(un_labelled_files_lst)):
    name = un_labelled_files_lst[i]
    print(name, counter)
    fold_name = 'case_' + str(counter)
    utils.makedir(os.path.join(data_path, fold_name))
    copyfile(os.path.join(un_labelled_path, name, 'img_left.npy'),
             os.path.join(data_path, fold_name, 'img_left.npy'))
    copyfile(os.path.join(un_labelled_path, name, 'img_right.npy'),
             os.path.join(data_path, fold_name, 'img_right.npy'))
    copyfile(os.path.join(un_labelled_path, name, 'segm_left.npy'),
             os.path.join(data_path, fold_name, 'segm_left.npy'))
    copyfile(os.path.join(un_labelled_path, name, 'segm_right.npy'),
             os.path.join(data_path, fold_name, 'segm_right.npy'))
    counter = counter + 1
