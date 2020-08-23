import os
import numpy as np
from utility.constants import *
from dataset_specific.prostate.utils.evaluation import generate_predictions
from utility.utils import makedir
from numpy.random import default_rng
from utility.config import get_metadata

def generate_uats_dataset(dataset_name, fold_num, labelled_perc, ul_imgs_path, supervised_model_path):

    metadata = get_metadata(dataset_name)
    unlabeled_imgs = np.load(ul_imgs_path)
    supervised_fold_path = os.path.join(metadata[m_data_path] , dataset_name , 'fold_' + str(fold_num)+'_P' + str(labelled_perc))
    labelled_num_considrd = len(os.listdir(os.path.join(supervised_fold_path, 'train', 'imgs')))
    counter = labelled_num_considrd
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

    print('copied labelled training images')
    for i in good_imgs_list:
        np.save(os.path.join(supervised_fold_path ,'train','imgs' , str(counter)) + '.npy', unlabeled_imgs[i])
        np.save(os.path.join(supervised_fold_path, 'train', 'gt' , str(counter)) + '.npy', pseudolabels[i])
        counter += 1
        print(i, counter)
    print('copied unlabelled training images')


def generate_supervised_dataset(dataset_name, fold_num, labelled_perc, seed=1234):

    metadata = get_metadata(dataset_name)
    folds_root_path = metadata[m_data_path]
    save_root_path = os.path.join(folds_root_path, dataset_name,'fold_' + str(fold_num) + '_P' + str(labelled_perc))
    supervised_fold_path = os.path.join(folds_root_path, dataset_name, 'fold_' + str(fold_num))
    labelled_num = metadata[m_labelled_train]
    # training
    makedir(os.path.join(save_root_path, 'train', 'imgs'), delete_existing=True)
    makedir(os.path.join(save_root_path, 'train', 'gt'), delete_existing=True)
    makedir(os.path.join(save_root_path, 'val', 'imgs'), delete_existing=True)
    makedir(os.path.join(save_root_path, 'val', 'gt'), delete_existing=True)
    num_labeled_train = int(labelled_perc * labelled_num)
    np.random.seed(seed)


    rng = default_rng()
    labelled_num_considrd = rng.choice(labelled_num, size=num_labeled_train, replace=False)
   # labelled_num_considrd = np.random.randint(0, labelled_num, size=num_labeled_train, dtype=np.int)

    counter = 0
    for i in labelled_num_considrd:
        np.save(os.path.join(save_root_path, 'train','imgs', str(counter)),
                np.load(os.path.join(supervised_fold_path, 'train', 'imgs', str(i) + '.npy')))
        np.save(os.path.join(save_root_path, 'train','gt', str(counter)),
                np.load(os.path.join(supervised_fold_path, 'train', 'gt',str(i) + '.npy')))
        counter = counter + 1
        print(i, counter)
    print('copied labelled training images')

    counter=0
    for i in np.arange(metadata[m_labelled_val]):
        np.save(os.path.join(save_root_path, 'val', 'imgs', str(counter)),
                np.load(os.path.join(supervised_fold_path, 'val', 'imgs', str(i) + '.npy')))
        np.save(os.path.join(save_root_path, 'val', 'gt', str(counter)),
                np.load(os.path.join(supervised_fold_path, 'val', 'gt', str(i) + '.npy')))
        print(i)
    print('copied labelled val images')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    ds = PROSTATE_DATASET_NAME
    metadata = get_metadata(ds)
    perc = 0.1
    fold_num = 3
    # generate_supervised_dataset(ds,
    #                             fold_num=fold_num,
    #                             labelled_perc=perc,
    #                             seed=0)

    generate_uats_dataset(ds,
        fold_num=fold_num,
        labelled_perc=perc,
        ul_imgs_path='/cache/suhita/data/' + ds + '/npy_img_unlabeled.npy',
        supervised_model_path=metadata[m_trained_model_path])