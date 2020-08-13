import os
import numpy as np
from utility.config import get_metadata
from utility.constants import *
from utility.prostate.evaluation import generate_predictions
from utility.utils import makedir


def generate_uats_dataset(dataset_name, fold_num, labelled_perc, ul_imgs_path, folds_root_path, supervised_model_path):
    # fold_num = 1
    # perc = 0.1
    # dataset_name = PROSTATE_DATASET_NAME
    # ul_imgs_path = '/cache/suhita/data/' + dataset_name + '/npy_img_unlabeled.npy'
    # folds_root_path ='/cache/suhita/data/'
    # supervised_model_path='/data/suhita/experiments/prostate/'
    metadata = get_metadata(dataset_name)
    dim = metadata[m_dim]
    unlabeled_imgs = np.load(ul_imgs_path)
    supervised_fold_path = folds_root_path + dataset_name + '/fold_' + str(fold_num) + '/'
    save_root_path = folds_root_path + dataset_name + '/'
    # generate predictions using suprvised model
    generate_predictions(supervised_model_path,
                         'supervised_F' + str(fold_num) + '_P' + str(labelled_perc),
                         unlabeled_imgs)
    unlabeled_imgs_gt = np.load(
        supervised_model_path + '/supervised_F' + str(fold_num) + '_P' + str(labelled_perc) + '.npy')
    # some unlabelled prostate images were excluded because they looked weird
    # filtering was done manually
    good_count = unlabeled_imgs.shape[0] - 8
    good_prediction_arr = np.empty((good_count, dim[0], dim[1], dim[2], metadata[m_nr_channels]))
    good_prediction_arr_gt = np.empty((good_count, dim[0], dim[1], dim[2], metadata[m_nr_class]))
    labelled_num = metadata[m_labelled_train]
    questionable = [47, 109, 203, 215]
    bad = [99, 100, 101, 103]
    all = np.arange(unlabeled_imgs.shape[0])
    conc = questionable + bad
    good_imgs_list = set(all) - set(conc)
    # training
    makedir(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/train/imgs/')
    makedir(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/train/gt/')
    makedir(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/val/imgs/')
    makedir(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/val/gt/')
    num_labeled_train = int(labelled_perc * labelled_num)
    labelled_num_considrd = [str(i) for i in np.arange(labelled_num)]
    np.random.seed(1234)
    np.random.shuffle(labelled_num_considrd)
    labelled_num_considrd = labelled_num_considrd[0:num_labeled_train]
    counter = 0
    for i in labelled_num_considrd:
        np.save(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/train/imgs/' + str(counter),
                np.load(os.path.join(supervised_fold_path, 'train', 'imgs', i + '.npy')))
        np.save(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/train/gt/' + str(counter),
                np.load(os.path.join(supervised_fold_path, 'train', 'gt', i + '.npy')))
        counter = counter + 1
        print(i, counter)
    print('copied labelled training images')
    for i in good_imgs_list:
        np.save(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/train/imgs/' + str(
            counter) + '.npy',
                unlabeled_imgs[i])
        np.save(
            save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/train/gt/' + str(counter) + '.npy',
            unlabeled_imgs_gt[i])
        counter += 1
        print(i, counter)
    print('copied unlabelled training images')
    for i in np.arange(metadata[m_labelled_val]):
        np.save(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/val/imgs/' + str(i),
                np.load(os.path.join(supervised_fold_path, 'val', 'imgs', str(i) + '.npy')))
        np.save(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/val/gt/' + str(i),
                np.load(os.path.join(supervised_fold_path, 'val', 'gt', str(i) + '.npy')))
        print(i)
    print('copied labelled val images')
