import os

import SimpleITK as sitk
import numpy as np

from utility.constants import *


def similarity3D_parameter_space_regular_sampling(thetaX, thetaY, thetaZ, tx, ty, tz, scale):
    '''
    Create a list representing a regular sampling of the 3D similarity transformation parameter space. As the
    SimpleITK rotation parameterization uses the vector portion of a versor we don't have an
    intuitive way of specifying rotations. We therefor use the ZYX Euler angle parametrization and convert to
    versor.
    Args:
        thetaX, thetaY, thetaZ: numpy ndarrays with the Euler angle values to use.
        tx, ty, tz: numpy ndarrays with the translation values to use.
        scale: numpy array with the scale values to use.
    Return:
        List of lists representing the parameter space sampling (vx,vy,vz,tx,ty,tz,s).
    '''
    return [list(eul2quat(parameter_values[0], parameter_values[1], parameter_values[2])) +
            [np.asscalar(p) for p in parameter_values[3:]] for parameter_values in
            np.nditer(np.meshgrid(thetaX, thetaY, thetaZ, tx, ty, tz, scale))]

def eul2quat(ax, ay, az, atol=1e-8):
    '''
    Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
    Args:
        ax: X rotation angle in radians.
        ay: Y rotation angle in radians.
        az: Z rotation angle in radians.
        atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    Return:
        Numpy array with three entries representing the vectorial component of the quaternion.

    '''
    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r = np.zeros((3, 3))
    r[0, 0] = cz * cy
    r[0, 1] = cz * sy * sx - sz * cx
    r[0, 2] = cz * sy * cx + sz * sx

    r[1, 0] = sz * cy
    r[1, 1] = sz * sy * sx + cz * cx
    r[1, 2] = sz * sy * cx - cz * sx

    r[2, 0] = -sy
    r[2, 1] = cy * sx
    r[2, 2] = cy * cx

    # Compute quaternion:
    qs = 0.5 * np.sqrt(r[0, 0] + r[1, 1] + r[2, 2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs, 0.0, atol):
        i = np.argmax([r[0, 0], r[1, 1], r[2, 2]])
        j = (i + 1) % 3
        k = (j + 1) % 3
        w = np.sqrt(r[i, i] - r[j, j] - r[k, k] + 1)
        qv[i] = 0.5 * w
        qv[j] = (r[i, j] + r[j, i]) / (2 * w)
        qv[k] = (r[i, k] + r[k, i]) / (2 * w)
    else:
        denom = 4 * qs
        qv[0] = (r[2, 1] - r[1, 2]) / denom;
        qv[1] = (r[0, 2] - r[2, 0]) / denom;
        qv[2] = (r[1, 0] - r[0, 1]) / denom;
    return qv

def augment_images_spatial(original_image, reference_image, T0, T_aug, transformation_parameters, flip_hor, flip_z,
                           output_prefix, output_suffix,
                           interpolator=sitk.sitkLinear, default_intensity_value=0.0, binary=False):
    # if binary:
    #     dist_filter = sitk.SignedMaurerDistanceMapImageFilter()
    #     dist_filter.SetUseImageSpacing(True)
    #     original_image = dist_filter.Execute(gt_bg)
    #     original_image = sitk.Cast(original_image, sitk.sitkFloat32)
    #     sitk.WriteImage(original_image, 'dist_map.nrrd')
    '''
    Generate the resampled images based on the given transformations.
    Args:
        original_image (SimpleITK image): The image which we will resample and transform.
        reference_image (SimpleITK image): The image onto which we will resample.
        T0 (SimpleITK transform): Transformation which maps points from the reference image coordinate system
            to the original_image coordinate system.
        T_aug (SimpleITK transform): Map points from the reference_image coordinate system back onto itself using the
               given transformation_parameters. The reason we use this transformation as a parameter
               is to allow the user to set its center of rotation to something other than zero.
        transformation_parameters (List of lists): parameter values which we use T_aug.SetParameters().
        output_prefix (string): output file name prefix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        output_suffix (string): output file name suffix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        interpolator: One of the SimpleITK interpolators.
        default_intensity_value: The value to return if a point is mapped outside the original_image domain.
    '''
    if flip_hor:
        arr = sitk.GetArrayFromImage(original_image)
        arr = np.flip(arr, axis=2)
        flipped = sitk.GetImageFromArray(arr)
        flipped.CopyInformation(original_image)
        original_image = flipped

    if flip_z:
        arr = sitk.GetArrayFromImage(original_image)
        arr = np.flip(arr, axis=0)
        flipped = sitk.GetImageFromArray(arr)
        flipped.CopyInformation(original_image)
        original_image = flipped

    for current_parameters in transformation_parameters:
        T_aug.SetParameters(current_parameters)
        # Augmentation is done in the reference image space, so we first map the points from the reference image space
        # back onto itself T_aug (e.g. rotate the reference image) and then we map to the original_classification image space T0.
        T_all = sitk.Transform(T0)
        T_all.AddTransform(T_aug)
        aug_image = sitk.Resample(original_image, reference_image, T_all,
                                  interpolator, default_intensity_value)

        # get distance image maximum
        # maxFilter = sitk.MinimumMaximumImageFilter()
        # maxFilter.Execute(aug_image)
        # minimum = maxFilter.GetMinimum()

        # if binary:
        #     sitk.WriteImage(aug_image, 'augm_img.nrrd')
        #     aug_image = sitk.BinaryThreshold(aug_image, minimum, 0)

        # arr = sitk.GetArrayFromImage(aug_image)
        # plt.imshow(arr[16,:,:], 'gray')
        # plt.savefig(output_prefix + '_' +
        #                     '_'.join(str(param) for param in current_parameters) + '.jpg')
        #

        # sitk.WriteImage(aug_image, output_prefix + '_' +
        #                     '_'.join(str(param) for param in current_parameters) + '.' + output_suffix)

        return aug_image


"""
def get_train_id_list(fold_num):
    if fold_num == 1:
        return np.arange(0, 58)
    elif fold_num == 2:
        return np.arange(20, 78)
    elif fold_num == 3:
        return np.concatenate((np.arange(40, 78), np.arange(0, 20)))
    elif fold_num == 4:
        return np.concatenate((np.arange(60, 78), np.arange(0, 40)))
    else:
        print('wrong fold number')
        return None
"""

def get_val_id_list(fold_num):
    if fold_num == 1:
        return np.arange(58, 78)
    elif fold_num == 2:
        return np.arange(0, 20)
    elif fold_num == 3:
        return np.arange(20, 40)
    elif fold_num == 4:
        return np.arange(40, 60)
    else:
        print('wrong fold number')
        return None


def get_val_data(data_path):
    val_fold = os.listdir(data_path[:-7] + VAL_IMGS_PATH)
    num_val_data = len(val_fold)
    val_supervised_flag = np.ones((num_val_data, PROSTATE_DIM[0], PROSTATE_DIM[1], PROSTATE_DIM[2]), dtype='int8')
    val_img_arr = np.zeros((num_val_data, PROSTATE_DIM[0], PROSTATE_DIM[1], PROSTATE_DIM[2], 1), dtype=float)
    val_GT_arr = np.zeros((num_val_data, PROSTATE_DIM[0], PROSTATE_DIM[1], PROSTATE_DIM[2], PROSTATE_NR_CLASS),
                          dtype=float)
    for i in np.arange(num_val_data):
        val_img_arr[i] = np.load(data_path[:-7] + VAL_IMGS_PATH + str(i) + NPY)
        val_GT_arr[i] = np.load(data_path[:-7] + VAL_GT_PATH + str(i) + NPY)
    x_val = [val_img_arr, val_GT_arr, val_supervised_flag]
    y_val = [val_GT_arr[:, :, :, :, 0], val_GT_arr[:, :, :, :, 1],
             val_GT_arr[:, :, :, :, 2], val_GT_arr[:, :, :, :, 3],
             val_GT_arr[:, :, :, :, 4]]
    return x_val, y_val


def get_uats_prostate_val_data(data_path):
    val_fold = os.listdir(data_path[:-7] + VAL_IMGS_PATH)
    num_val_data = len(val_fold)
    val_supervised_flag = np.ones((num_val_data, PROSTATE_DIM[0], PROSTATE_DIM[1], PROSTATE_DIM[2]), dtype='int64')
    val_img_arr = np.zeros((num_val_data, PROSTATE_DIM[0], PROSTATE_DIM[1], PROSTATE_DIM[2], 1), dtype=float)
    val_GT_arr = np.zeros((num_val_data, PROSTATE_DIM[0], PROSTATE_DIM[1], PROSTATE_DIM[2], PROSTATE_NR_CLASS),
                          dtype=float)
    for i in np.arange(num_val_data):
        val_img_arr[i] = np.load(data_path[:-7] + VAL_IMGS_PATH + str(i) + NPY)
        val_GT_arr[i] = np.load(data_path[:-7] + VAL_GT_PATH + str(i) + NPY)
    x_val = [val_img_arr, val_GT_arr, val_supervised_flag]
    y_val = [val_GT_arr[:, :, :, :, 0], val_GT_arr[:, :, :, :, 1],
             val_GT_arr[:, :, :, :, 2], val_GT_arr[:, :, :, :, 3],
             val_GT_arr[:, :, :, :, 4]]
    return x_val, y_val
