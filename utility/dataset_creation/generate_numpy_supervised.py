import os
import numpy as np
from utility.config import get_metadata
from utility.constants import *
from utility.utils import makedir
from numpy.random import default_rng


def generate_supervised_dataset(dataset_name, fold_num, labelled_perc, folds_root_path, seed=1234):

    metadata = get_metadata(dataset_name)
    supervised_fold_path = folds_root_path + dataset_name + '/fold_' + str(fold_num) + '/'
    save_root_path = folds_root_path + dataset_name + '/'
    labelled_num = metadata[m_labelled_train]
    # training
    makedir(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/train/imgs/', delete_existing=True)
    makedir(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/train/gt/', delete_existing=True)
    makedir(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/val/imgs/', delete_existing=True)
    makedir(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/val/gt/', delete_existing=True)
    num_labeled_train = int(labelled_perc * labelled_num)
    np.random.seed(seed)


    rng = default_rng()
    labelled_num_considrd = rng.choice(labelled_num, size=num_labeled_train, replace=False)
   # labelled_num_considrd = np.random.randint(0, labelled_num, size=num_labeled_train, dtype=np.int)

    counter = 0
    for i in labelled_num_considrd:
        np.save(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/train/imgs/' + str(counter),
                np.load(os.path.join(supervised_fold_path, 'train', 'imgs', str(i) + '.npy')))
        np.save(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/train/gt/' + str(counter),
                np.load(os.path.join(supervised_fold_path, 'train', 'gt',str(i) + '.npy')))
        counter = counter + 1
        print(i, counter)
    print('copied labelled training images')

    for i in np.arange(metadata[m_labelled_val]):
        np.save(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/val/imgs/' + str(i),
                np.load(os.path.join(supervised_fold_path, 'val', 'imgs', str(i) + '.npy')))
        np.save(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/val/gt/' + str(i),
                np.load(os.path.join(supervised_fold_path, 'val', 'gt', str(i) + '.npy')))
        print(i)
    print('copied labelled val images')
