import os
from shutil import copyfile
import numpy as np
from utility.config import get_metadata
from utility.constants import *
from utility.utils import makedir
from utility.skin.preprocess import thresholdArray, castImage, getLargestConnectedComponents, getConnectedComponents
import SimpleITK as sitk


def generate_supervised_dataset(dataset_name, fold_num, labelled_perc, seed=1234):

    metadata = get_metadata(dataset_name)
    save_root_path = os.path.join(metadata[m_data_path] , dataset_name )
    supervised_fold_path = os.path.join(save_root_path ,'fold_' + str(fold_num) + '_P' + str(labelled_perc))

    labelled_files_lst = np.load(os.path.join(metadata[m_folds_path], 'train_fold'+str(fold_num)+'.npy'))
    labelled_train_num = len(labelled_files_lst)
    labelled_path = metadata[m_raw_data_path] +'/labelled/train/'
    print(labelled_files_lst[0:10])
    np.random.seed(seed)
    np.random.shuffle(labelled_files_lst)

    labelled_num_considrd = labelled_files_lst[:int(labelled_train_num * labelled_perc)]
    validation_imgs = np.load(os.path.join(metadata[m_folds_path], 'val_fold' + str(fold_num) + '.npy'))

    counter = 0
    makedir( os.path.join(supervised_fold_path,  'train','imgs'), delete_existing=True)
    makedir( os.path.join(supervised_fold_path,  'train','gt'), delete_existing=True)
    makedir( os.path.join(supervised_fold_path,  'val','imgs'), delete_existing=True)
    makedir( os.path.join(supervised_fold_path,  'val','gt'), delete_existing=True)

    print('training images created')
    for i in labelled_num_considrd:
        print(i, counter)
        np.save(os.path.join(supervised_fold_path, 'train', 'imgs', str(counter) + '.npy'),
                np.load(os.path.join(labelled_path, 'imgs', i)) / 255
                )
        GT_lesion = np.load(os.path.join(labelled_path, 'GT', i.replace('.npy', '_segmentation.npy'))) / 255
        GT_bg = np.where(GT_lesion == 0, np.ones_like(GT_lesion), np.zeros_like(GT_lesion))

        np.save(os.path.join(supervised_fold_path, 'train', 'gt', str(counter) + '.npy'),
                np.concatenate((GT_bg, GT_lesion), -1))
        counter = counter + 1

    print('validation images created')
    counter = 0
    for i in validation_imgs:
        print(i, counter)
        np.save(os.path.join(supervised_fold_path, 'val', 'imgs', str(counter) + '.npy'),
                np.load(os.path.join(labelled_path, 'imgs', i)) / 255)
        GT_lesion = np.load(os.path.join(labelled_path, 'GT', i.replace('.npy', '_segmentation.npy'))) / 255
        GT_bg = np.where(GT_lesion == 0, np.ones_like(GT_lesion), np.zeros_like(GT_lesion))

        np.save(os.path.join(supervised_fold_path, 'val', 'gt', str(counter) + '.npy'),
                np.concatenate((GT_bg, GT_lesion), -1))
        counter = counter + 1

def generate_uats_dataset(dataset_name, fold_num, labelled_perc, ul_imgs_path, supervised_model_path):

    metadata = get_metadata(dataset_name)
    #unlabeled_imgs = np.load(ul_imgs_path)
    dim = metadata[m_dim]
    nr_channels = metadata[m_nr_channels]
    supervised_fold_path = os.path.join(metadata[m_data_path] , dataset_name , 'fold_' + str(fold_num)+'_P' + str(labelled_perc))
    labelled_num_considrd = len(os.listdir(os.path.join(supervised_fold_path, 'train', 'imgs')))
    counter = labelled_num_considrd
    # generate predictions using suprvised model
    cases = sorted(os.listdir(os.path.join(ul_imgs_path)))
    img_arr = np.zeros((len(cases),dim[0],dim[1], nr_channels), dtype=float)
    for i in range(len(cases)):
        img_arr[i] = np.load(os.path.join(ul_imgs_path, cases[i])) / 255
    print(img_arr.shape[0],' unlabelled images loaded')

    from dataset_specific.skin_2D.model.baseline import weighted_model
    wm = weighted_model()
    model = wm.build_model(img_shape=(dim[0], dim[1], nr_channels), learning_rate=5e-05)
    model.load_weights(supervised_model_path)
    prediction = model.predict(img_arr, batch_size=1)
    nrImgs = prediction[0].shape[0]
    for imgNumber in range(0, nrImgs):
        prediction_skin = prediction[1]
        pred_arr = prediction_skin[imgNumber]
        pred_arr = thresholdArray(pred_arr, 0.5)
        pred_img = sitk.GetImageFromArray(pred_arr)  # only foreground=skin for evaluation
        pred_img = castImage(pred_img, sitk.sitkUInt8)

        prediction_bg = prediction[0]
        pred_arr_bg = prediction_bg[imgNumber]
        pred_arr_bg = thresholdArray(pred_arr_bg, 0.5)

        pred_img = getConnectedComponents(pred_img)
        pred_img_arr = sitk.GetArrayFromImage(pred_img)
        GT_out = np.zeros([pred_arr.shape[0], pred_arr.shape[1], 2])
        GT_out[:, :, 0] = pred_arr_bg
        GT_out[:, :, 1] = pred_arr

        np.save(os.path.join(supervised_fold_path, 'train', 'imgs', str(counter)) + '.npy', img_arr[imgNumber])
        np.save(os.path.join(supervised_fold_path, 'train', 'gt', str(counter)) + '.npy', GT_out)
        counter+=1
        print(i, counter)
    print('copied unlabelled training images')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    generate_supervised_dataset(SKIN_DATASET_NAME, 1, 1.0, seed=1234)
    ul_path = '/cache/suhita/skin/preprocessed/unlabelled/imgs'
    generate_uats_dataset(SKIN_DATASET_NAME, 1, 1.0, ul_path,
                          '/data/suhita/experiments/model/supervised/skin/supervised_F1_P1.0.h5')
