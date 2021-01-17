import numpy as np
import os
import shutil

import SimpleITK as sitk
import numpy as np

from dataset_specific.kits import utils

def get_multi_class_arr(arr, n_classes=3):
    size = arr.shape
    out_arr = np.zeros([size[0], size[1], size[2], n_classes])

    for i in range(n_classes):
        arr_temp = arr.copy()
        out_arr[:, :, :, i] = np.where(arr_temp == i, 1, 0)
        del arr_temp
    return out_arr

def normalizeIntensities(*imgs):
    out = []

    for img in imgs:
        array = np.ndarray.flatten(sitk.GetArrayFromImage(img))

        upperPerc = np.percentile(array, 100)  # 98
        lowerPerc = np.percentile(array, 0)  # 2
        print('percentiles', upperPerc, lowerPerc)

        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
        normalizationFilter = sitk.IntensityWindowingImageFilter()
        normalizationFilter.SetOutputMaximum(1.0)
        normalizationFilter.SetOutputMinimum(0.0)
        normalizationFilter.SetWindowMaximum(upperPerc)
        normalizationFilter.SetWindowMinimum(lowerPerc)

        floatImg = castImageFilter.Execute(img)
        outNormalization = normalizationFilter.Execute(floatImg)
        out.append(outNormalization)

    return out


def create_and_move_test_images(in_dir, out_dir, nr_test_images):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = os.listdir(in_dir)
    np.random.seed(1234)
    files_idx = np.arange(0, len(files))
    np.random.shuffle(files_idx)

    for file_ix in files_idx[0:nr_test_images]:
        filename = files[file_ix]
        shutil.move(os.path.join(in_dir, files[file_ix]), os.path.join(out_dir, filename))


def preprocess_dir(in_dir, out_dir, GT=False):
    utils.makeDirectory(out_dir)

    cases = os.listdir(in_dir)

    # sizes = np.zeros([len(cases) - 1, 3])
    i = 0

    for case in cases:
        if not case.startswith('.'):

            img = sitk.ReadImage(os.path.join(in_dir, case))
            # label = sitk.ReadImage(os.path.join(train_dir, case))
            if not GT:
                img = normalizeIntensities(img)[0]

            img = utils.pad_volume(img, target_size_x=48, target_size_y=64, target_size_z=48, padValue=0)
            filename = case[:-7]
            sitk.WriteImage(img, os.path.join(out_dir, filename + '.nrrd'))
            np.save(os.path.join(out_dir, filename + '.npy'), sitk.GetArrayFromImage(img))


if __name__ == '__main__':
    in_dir = '/data/anneke/hippocampus/labelled/train'
    in_dir_label = '/data/anneke/hippocampus/labelled_GT/train'
    out_dir = '/data/anneke/hippocampus/labelled_GT/test'
    # create_and_move_test_images(in_dir, out_dir, nr_test_images=50)

    # cases = os.listdir(in_dir)
    # for case in cases:
    #
    #     shutil.move(os.path.join(in_dir_label, case), os.path.join(out_dir, case))

    # preprocess_dir(in_dir= '/data/anneke/hippocampus/labelled-GT/train',
    #                out_dir= '/data/anneke/hippocampus/preprocessed/labelled-GT/train', GT=True)
    # preprocess_dir(in_dir='/data/anneke/hippocampus/labelled/test',
    #                out_dir='/data/anneke/hippocampus/preprocessed/labelled/test', GT=False)

    utils.generateFolds(in_dir, 'Folds', 4)