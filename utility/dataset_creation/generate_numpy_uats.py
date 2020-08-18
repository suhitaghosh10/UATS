import os
import numpy as np
from utility.config import get_metadata
from utility.constants import *
from utility.prostate.evaluation import generate_predictions
from utility.utils import makedir


def generate_uats_dataset(dataset_name, fold_num, labelled_perc, ul_imgs_path, folds_root_path, supervised_model_path):

    unlabeled_imgs = np.load(ul_imgs_path)
    supervised_fold_path = os.path.join(folds_root_path , dataset_name , 'fold_' + str(fold_num))
    #save_root_path = folds_root_path + dataset_name + '/'
    # generate predictions using suprvised model
    pseudolabels = generate_predictions(os.path.join(supervised_model_path,
                                      dataset_name),
                                      'supervised_F' + str(fold_num) + '_P' + str(labelled_perc),
                         unlabeled_imgs)
    # some unlabelled prostate images were excluded because they looked weird
    # filtering was done manually
    questionable = [47, 109, 203, 215]
    bad = [99, 100, 101, 103]
    all = np.arange(unlabeled_imgs.shape[0])
    conc = questionable + bad
    good_imgs_list = set(all) - set(conc)
    labelled_num_considrd = len(os.path.join(supervised_fold_path, 'train', 'imgs'))
    counter = labelled_num_considrd
    # for i in labelled_num_considrd:
    #     np.save(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/train/imgs/' + str(counter),
    #             np.load(os.path.join(supervised_fold_path, 'train', 'imgs', i + '.npy')))
    #     np.save(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/train/gt/' + str(counter),
    #             np.load(os.path.join(supervised_fold_path, 'train', 'gt', i + '.npy')))
    #     counter = counter + 1
    #     print(i, counter)
    print('copied labelled training images')
    for i in good_imgs_list:
        np.save(os.path.join(supervised_fold_path , '_P' + str(labelled_perc) , 'train','imgs' + str(
            counter)) + '.npy',
                unlabeled_imgs[i])
        np.save(
            os.path.join(supervised_fold_path, '_P' + str(labelled_perc), 'train', 'gt' + str(
                counter)) + '.npy',
            pseudolabels[i])
        counter += 1
        print(i, counter)
    print('copied unlabelled training images')
    # for i in np.arange(metadata[m_labelled_val]):
    #     np.save(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/val/imgs/' + str(i),
    #             np.load(os.path.join(supervised_fold_path, 'val', 'imgs', str(i) + '.npy')))
    #     np.save(save_root_path + 'fold_' + str(fold_num) + '_P' + str(labelled_perc) + '/val/gt/' + str(i),
    #             np.load(os.path.join(supervised_fold_path, 'val', 'gt', str(i) + '.npy')))
    #     print(i)
    # print('copied labelled val images')
