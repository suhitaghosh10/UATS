import os
import random as rn
from enum import Enum

import SimpleITK as sitk
import numpy as np
from keras.utils.np_utils import to_categorical

rn.seed(1235)
write_flag = False

OUTPUT_DIR = '/cache/anneke/uats'

# f = open('/cache/anneke/uats/flag_afs_nr.csv', 'w')
# csvwriter = csv.writer(f, delimiter=';', lineterminator='\n',
#                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

reference_size = [168, 168, 32]
reference_spacing = [0.5, 0.5, 3.0]
dimension = 3
gt_shape = [32, 168, 168, 5]
zones_num = 5


class AugmentTypes(Enum):
    ROTATE = 0
    FLIP_HORIZ = 1
    TRANSLATION_3D = 2
    SCALE = 3
    ORIGINAL = 4


def resampleImage(inputImage, newSpacing, interpolator, defaultValue):
    # castImageFilter = sitk.CastImageFilter()
    # castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    # inputImage = castImageFilter.Execute(inputImage)

    oldSize = inputImage.GetSize()
    oldSpacing = inputImage.GetSpacing()
    newWidth = oldSpacing[0] / newSpacing[0] * oldSize[0]
    newHeight = oldSpacing[1] / newSpacing[1] * oldSize[1]
    newDepth = oldSpacing[2] / newSpacing[2] * oldSize[2]
    newSize = [int(newWidth), int(newHeight), int(newDepth)]

    # minFilter = sitk.StatisticsImageFilter()
    # minFilter.Execute(inputImage)
    # minValue = minFilter.GetMinimum()

    filter = sitk.ResampleImageFilter()
    inputImage.GetSpacing()
    filter.SetOutputSpacing(newSpacing)
    filter.SetInterpolator(interpolator)
    filter.SetOutputOrigin(inputImage.GetOrigin())
    filter.SetOutputDirection(inputImage.GetDirection())
    filter.SetSize(newSize)
    filter.SetDefaultPixelValue(defaultValue)
    outImage = filter.Execute(inputImage)

    return outImage


def resampleToReference(inputImg, referenceImg, interpolator, defaultValue):
    # castImageFilter = sitk.CastImageFilter()
    # castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    # inputImg = castImageFilter.Execute(inputImg)

    # minFilter = sitk.StatisticsImageFilter()
    # minFilter.Execute(inputImg)

    filter = sitk.ResampleImageFilter()
    filter.SetReferenceImage(referenceImg)
    filter.SetDefaultPixelValue(float(defaultValue))  ## -1
    # float('nan')

    filter.SetInterpolator(interpolator)
    outImage = filter.Execute(inputImg)

    return outImage


def augment_images_spatial(original_image, reference_image, augmentation_type, T0, T_aug, transformation_parameters,
                           interpolator=sitk.sitkLinear, default_intensity_value=0.0):
    # interpolator = sitk.sitkNearestNeighbor
    if augmentation_type == AugmentTypes.FLIP_HORIZ.value:
        arr = sitk.GetArrayFromImage(original_image)
        arr = np.flip(arr, axis=2)
        aug_image = sitk.GetImageFromArray(arr)
        aug_image.CopyInformation(original_image)

    else:
        for current_parameters in transformation_parameters:
            T_aug.SetParameters(current_parameters)
            # Augmentation is done in the reference image space, so we first map the points from the reference image space
            # back onto itself T_aug (e.g. rotate the reference image) and then we map to the original_classification image space T0.
            T_all = sitk.Transform(T0)
            T_all.AddTransform(T_aug)
            aug_image = sitk.Resample(original_image, reference_image, T_all,
                                      interpolator, default_intensity_value)
            aug_image = resampleToReference(aug_image, reference_image, interpolator, default_intensity_value)

    return aug_image


def check_if_doubles(rand_vectors, new_vector):
    for vec in rand_vectors:
        if new_vector == vec:
            return True
        else:
            return False


def write_image(image, image_name, ref_image, is_image=False):
    if (write_flag):
        writer = sitk.ImageFileWriter()
        writer.SetFileName(image_name)
        if not is_image:
            temp = sitk.GetImageFromArray(image)
            temp.SetSpacing(reference_spacing)
        else:
            temp = image

        # temp.CopyInformation(ref_image)
        writer.Execute(temp)


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


def get_reference_image(image):
    # Create the reference image with a zero origin, identity direction cosine matrix and dimension

    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_image = sitk.Image(reference_size, image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    return reference_image


def get_augmentation_transform(img, reference_image, augmentation_type):
    aug_transform = sitk.Similarity2DTransform() if dimension == 2 else sitk.Similarity3DTransform()
    reference_origin = np.zeros(dimension)
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)

    # Modify the transformation to align the centers of the original_classification and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    # Set the augmenting transform's center so that rotation is around the image center.
    aug_transform.SetCenter(reference_center)

    delta_Arr = [np.random.uniform(-0.174533, 0.174533), np.random.uniform(-0.174533, 0.174533), 0.0,
                 np.random.uniform(-0.174533, 0.174533), np.random.uniform(-0.174533, 0.174533)]
    translationsXY_Arr = [np.random.uniform(-4, 4), np.random.uniform(-4, 4), 0.0, np.random.uniform(-4, 4),
                          np.random.uniform(-4, 4)]  # in mm
    translationsZ_Arr = [np.random.uniform(-6, 6), np.random.uniform(-6, 6), 0.0, np.random.uniform(-6, 6),
                         np.random.uniform(-6, 6)]
    scale_factor = np.random.uniform(0.9, 1.1)
    # vecs = []
    rand_vec = np.array(
        [rn.randint(0, 4), rn.randint(0, 4), rn.randint(1, 3), rn.randint(0, 4), rn.randint(0, 4), rn.randint(0, 4),
         rn.randint(0, 2), rn.randint(0, 1)], dtype=int)

    # print(rand_vec)
    if AugmentTypes.ROTATE.value == augmentation_type:
        delta_x = delta_Arr[rand_vec[0]]
        delta_y = delta_Arr[rand_vec[1]]
        delta_z = delta_Arr[rand_vec[2]]
        transl_x = translationsXY_Arr[rand_vec[3]]
        transl_y = translationsXY_Arr[rand_vec[4]]
        transl_z = translationsZ_Arr[rand_vec[5]]
        scale = 1.

    elif AugmentTypes.SCALE.value == augmentation_type:
        delta_x = 0.
        delta_y = 0
        delta_z = 0.
        transl_x = 0.
        transl_y = 0.
        transl_z = 0.
        scale = scale_factor

    elif AugmentTypes.TRANSLATION_3D.value == augmentation_type:
        delta_x = 0.
        delta_y = 0
        delta_z = 0.
        transl_x = translationsXY_Arr[rand_vec[3]]
        transl_y = translationsXY_Arr[rand_vec[4]]
        transl_z = translationsZ_Arr[rand_vec[5]]
        scale = 1.
    else:
        delta_x = 0.
        delta_y = 0
        delta_z = 0.
        transl_x = 0.
        transl_y = 0.
        transl_z = 0.
        scale = 1.

    transformation_parameters_list = similarity3D_parameter_space_regular_sampling([delta_x], [delta_y], [delta_z],
                                                                                   [transl_x], [transl_y],
                                                                                   [transl_z], [scale])

    return centered_transform, aug_transform, transformation_parameters_list


def get_transformed_gt(orig_gt, augmentation_type, centered_transform, aug_transform, transformation_parameters_list):
    orig_gt = np.where(orig_gt > 0.5, np.ones_like(orig_gt), np.zeros_like(orig_gt))
    orig_gt = orig_gt.astype('int64')
    gt_distances = np.zeros(gt_shape)

    for zone in range(0, zones_num):
        orig_img_gt = sitk.GetImageFromArray(orig_gt[:, :, :, zone])
        orig_img_gt.SetSpacing(reference_spacing)

        write_image(orig_img_gt, os.path.join(OUTPUT_DIR, 'orig_gt' + str(zone) + '.nrrd'), orig_img_gt, is_image=True)

        gt_dist = sitk.SignedMaurerDistanceMap(orig_img_gt, insideIsPositive=True, squaredDistance=False,
                                               useImageSpacing=True)

        resampled_dist = augment_images_spatial(gt_dist, orig_img_gt, augmentation_type, centered_transform,
                                                aug_transform, transformation_parameters_list,
                                                default_intensity_value=-3000,
                                                interpolator=sitk.sitkLinear)

        gt_distances[:, :, :, zone] = sitk.GetArrayFromImage(resampled_dist)
    background_arr = np.zeros(gt_distances.shape, dtype=np.uint8)
    background_arr[:, :, :, 4] = 1
    # get indices where distances are maxmimum (consider last axis)
    gt_distances = np.where(gt_distances > -3000, gt_distances, background_arr)
    indices = np.argmax(gt_distances, axis=3)
    # indices to one-hot vector
    res_gt = to_categorical(indices, num_classes=5)
    res_gt.astype(np.uint8)

    return res_gt


def get_transformed_flag(orig_flag, augmentation_type, centered_transform, aug_transform,
                         transformation_parameters_list):
    # orig_gt = np.where(orig_gt > 0.5, np.ones_like(orig_gt), np.zeros_like(orig_gt))
    # miximaxi = np.min(orig_flag)
    # miximini = np.max(orig_flag)

    # csvwriter.writerow(['initial', miximini, miximaxi])

    orig_flag = orig_flag.astype('int64')
    # miximaxi = np.max(orig_flag)
    # miximini = np.max(orig_flag)
    # csvwriter.writerow(['after type conversion', miximini, miximaxi])

    # gt_distances = np.zeros((gt_shape[0], gt_shape[1], gt_shape[2], 2))

    orig_img_gt = sitk.GetImageFromArray(orig_flag)
    orig_img_gt.SetSpacing(reference_spacing)

    write_image(orig_img_gt, os.path.join(OUTPUT_DIR, 't-orig_flag.nrrd'), orig_img_gt,
                is_image=True)

    gt_dist = sitk.SignedMaurerDistanceMap(orig_img_gt, insideIsPositive=True, squaredDistance=False,
                                           useImageSpacing=True)

    resampled_dist = augment_images_spatial(gt_dist, orig_img_gt, augmentation_type, centered_transform,
                                            aug_transform, transformation_parameters_list,
                                            default_intensity_value=-3000,
                                            interpolator=sitk.sitkLinear)

    gt_distances = sitk.GetArrayFromImage(resampled_dist)
    flag = np.where(gt_distances >= 0, 1, 0)
    return flag


def get_single_image_augmentation_with_ensemble(augmentation_type, orig_image, orig_gt, ens_gt, flag, img_no,
                                                labelled_num):
    # print(img_no, augmentation_type)
    out_img = np.zeros([32, 168, 168, 1], dtype=np.float32)

    img1 = sitk.GetImageFromArray(orig_image[:, :, :, 0])
    img1.SetSpacing(reference_spacing)
    reference_image = get_reference_image(img1)

    img = sitk.GetImageFromArray(orig_image)
    img.SetSpacing(reference_spacing)
    write_image(img, os.path.join(OUTPUT_DIR, 'orig_image' + str(img_no) + '.nrrd'), reference_image, is_image=True)

    centered_transform, aug_transform, transformation_parameters_list = get_augmentation_transform(img, reference_image,
                                                                                                   augmentation_type)

    # transform image
    res_img = augment_images_spatial(img, reference_image, augmentation_type, centered_transform,
                                     aug_transform, transformation_parameters_list)

    out_img[:, :, :, 0] = sitk.GetArrayFromImage(res_img)

    # transform gt
    gt_ref = sitk.GetImageFromArray(orig_gt)
    gt_ref.SetSpacing(reference_spacing)
    write_image(res_img, os.path.join(OUTPUT_DIR, 'changed_image' + str(img_no) + '_' + AugmentTypes(
        augmentation_type).name + '.nrrd'), reference_image, is_image=True)

    # out_gt = get_transformed_ens_gt2(orig_gt, augmentation_type, centered_transform, aug_transform, transformation_parameters_list)
    if int(img_no) < labelled_num:
        out_gt = get_transformed_gt(orig_gt, augmentation_type, centered_transform, aug_transform,
                                    transformation_parameters_list)
        # ch_flag = np.where(flag >= 1, 1, 0)
        # ch_flag = get_transformed_flag(ch_flag, augmentation_type, centered_transform, aug_transform,
        #                             transformation_parameters_list)
        ch_flag = sitk.GetImageFromArray(flag)
        ch_flag.SetSpacing(reference_spacing)
        ch_flag = augment_images_spatial(ch_flag, reference_image, augmentation_type, centered_transform,
                                         aug_transform, transformation_parameters_list,
                                         interpolator=sitk.sitkNearestNeighbor, default_intensity_value=1.0)
        write_image(img, os.path.join(OUTPUT_DIR, 'ch_flag' + str(img_no) + '.nrrd'), reference_image, is_image=True)

        ch_flag = sitk.GetArrayFromImage(ch_flag)
        ch_flag = np.where(ch_flag > 0.5, 1, 0)


    else:
        out_gt = augment_images_spatial(gt_ref, reference_image, augmentation_type, centered_transform,
                                        aug_transform, transformation_parameters_list)
        out_gt = sitk.GetArrayFromImage(out_gt)

        # flag_img = sitk.GetImageFromArray(flag)
        # flag_img.SetSpacing(reference_spacing)
        ch_flag = np.where(flag >= 1, 1, 0)

        # flag_img = sitk.GetImageFromArray(ch_flag)
        # flag_img.SetSpacing(reference_spacing)
        # ch_flag = augment_images_spatial(flag_img, reference_image, augmentation_type, centered_transform,
        #                                  aug_transform, transformation_parameters_list,
        #                                  interpolator=sitk.sitkNearestNeighbor)
        # ch_flag = sitk.GetArrayFromImage(ch_flag)

        ch_flag = get_transformed_flag(ch_flag, augmentation_type, centered_transform, aug_transform,
                                       transformation_parameters_list)
        # flaggi = sitk.GetImageFromArray(ch_flag)
        # flaggi.SetSpacing(reference_spacing)
        # sitk.WriteImage(flaggi, os.path.join(OUTPUT_DIR, 'ch_flag' + str(img_no) + '.nrrd'))

        ch_flag = np.where(ch_flag > 0.5, 2, 0)

    write_image(flag.astype('int64'),
                os.path.join(OUTPUT_DIR, 'orig_flag' + str(img_no) + '_' + str(0) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref,
                is_image=False)
    # write_image(ch_flag, os.path.join(OUTPUT_DIR, 'ch_flag' + str(img_no) + '_' + str(0) + '_' + AugmentTypes(
    #     augmentation_type).name + '.nrrd'), gt_ref,
    #             is_image=False)
    write_image(out_gt[:, :, :, 0],
                os.path.join(OUTPUT_DIR, 'ch_gt' + str(img_no) + '_' + str(0) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_gt[:, :, :, 1],
                os.path.join(OUTPUT_DIR, 'ch_gt' + str(img_no) + '_' + str(1) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_gt[:, :, :, 2],
                os.path.join(OUTPUT_DIR, 'ch_gt' + str(img_no) + '_' + str(2) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_gt[:, :, :, 3],
                os.path.join(OUTPUT_DIR, 'ch_gt' + str(img_no) + '_' + str(3) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_gt[:, :, :, 4],
                os.path.join(OUTPUT_DIR, 'ch_gt' + str(img_no) + '_' + str(4) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)

    # out_ens_gt = get_transformed_ens_gt2(ens_gt, augmentation_type, centered_transform, aug_transform,transformation_parameters_list)
    out_ens_gt = sitk.GetImageFromArray(ens_gt)
    out_ens_gt.SetSpacing(reference_spacing)
    out_ens_gt = augment_images_spatial(out_ens_gt, reference_image, augmentation_type, centered_transform,
                                        aug_transform, transformation_parameters_list, default_intensity_value=-1.0)
    out_ens_gt = sitk.GetArrayFromImage(out_ens_gt)
    out_ens_gt[:, :, :, 0:4] = np.where(out_ens_gt[:, :, :, 0:4] == -1, np.zeros_like(out_ens_gt[:, :, :, 0:4]),
                                        out_ens_gt[:, :, :, 0:4])
    out_ens_gt[:, :, :, 4] = np.where(out_ens_gt[:, :, :, 4] == -1, np.ones_like(out_ens_gt[:, :, :, 4]),
                                      out_ens_gt[:, :, :, 4])

    write_image(out_ens_gt[:, :, :, 0],
                os.path.join(OUTPUT_DIR, 'ch_ens_gt' + str(img_no) + '_' + str(0) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_ens_gt[:, :, :, 1],
                os.path.join(OUTPUT_DIR, 'ch_ens_gt' + str(img_no) + '_' + str(1) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_ens_gt[:, :, :, 2],
                os.path.join(OUTPUT_DIR, 'ch_ens_gt' + str(img_no) + '_' + str(2) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_ens_gt[:, :, :, 3],
                os.path.join(OUTPUT_DIR, 'ch_ens_gt' + str(img_no) + '_' + str(3) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_ens_gt[:, :, :, 4],
                os.path.join(OUTPUT_DIR, 'ch_ens_gt' + str(img_no) + '_' + str(4) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)

    return out_img, out_gt, out_ens_gt, ch_flag


def get_single_image_augmentation(augmentation_type, orig_image, orig_gt, img_no):
    out_img = np.zeros([32, 168, 168, 1], dtype=np.float32)

    img1 = sitk.GetImageFromArray(orig_image[:, :, :, 0])
    img1.SetSpacing(reference_spacing)
    reference_image = get_reference_image(img1)

    img = sitk.GetImageFromArray(orig_image)
    img.SetSpacing(reference_spacing)

    centered_transform, aug_transform, transformation_parameters_list = get_augmentation_transform(img, reference_image,
                                                                                                   augmentation_type)

    # transform image
    res_img = augment_images_spatial(img, reference_image, augmentation_type, centered_transform,
                                     aug_transform, transformation_parameters_list)

    out_img[:, :, :, 0] = sitk.GetArrayFromImage(res_img)

    # transform gt
    gt_ref = sitk.GetImageFromArray(orig_gt)
    gt_ref.SetSpacing(reference_spacing)

    out_gt = get_transformed_gt(orig_gt, augmentation_type, centered_transform, aug_transform,
                                transformation_parameters_list)

    return out_img, out_gt


def get_single_image_augmentation_with_only_ensemble(augmentation_type, orig_image, orig_gt, ens_gt, img_no,
                                                     labelled_num):
    # print(img_no, augmentation_type)
    out_img = np.zeros([32, 168, 168, 1], dtype=np.float32)

    img1 = sitk.GetImageFromArray(orig_image[:, :, :, 0])
    img1.SetSpacing(reference_spacing)
    reference_image = get_reference_image(img1)

    img = sitk.GetImageFromArray(orig_image)
    img.SetSpacing(reference_spacing)
    write_image(img, os.path.join(OUTPUT_DIR, 'orig_image' + str(img_no) + '.nrrd'), reference_image, is_image=True)

    centered_transform, aug_transform, transformation_parameters_list = get_augmentation_transform(img, reference_image,
                                                                                                   augmentation_type)

    # transform image
    res_img = augment_images_spatial(img, reference_image, augmentation_type, centered_transform,
                                     aug_transform, transformation_parameters_list)

    out_img[:, :, :, 0] = sitk.GetArrayFromImage(res_img)

    # transform gt
    gt_ref = sitk.GetImageFromArray(orig_gt)
    gt_ref.SetSpacing(reference_spacing)
    write_image(res_img, os.path.join(OUTPUT_DIR, 'changed_image' + str(img_no) + '_' + AugmentTypes(
        augmentation_type).name + '.nrrd'), reference_image, is_image=True)

    # out_gt = get_transformed_ens_gt2(orig_gt, augmentation_type, centered_transform, aug_transform, transformation_parameters_list)
    if int(img_no) < labelled_num:
        out_gt = get_transformed_gt(orig_gt, augmentation_type, centered_transform, aug_transform,
                                    transformation_parameters_list)

    else:
        out_gt = augment_images_spatial(gt_ref, reference_image, augmentation_type, centered_transform,
                                        aug_transform, transformation_parameters_list)
        out_gt = sitk.GetArrayFromImage(out_gt)

    write_image(out_gt[:, :, :, 0],
                os.path.join(OUTPUT_DIR, 'ch_gt' + str(img_no) + '_' + str(0) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_gt[:, :, :, 1],
                os.path.join(OUTPUT_DIR, 'ch_gt' + str(img_no) + '_' + str(1) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_gt[:, :, :, 2],
                os.path.join(OUTPUT_DIR, 'ch_gt' + str(img_no) + '_' + str(2) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_gt[:, :, :, 3],
                os.path.join(OUTPUT_DIR, 'ch_gt' + str(img_no) + '_' + str(3) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_gt[:, :, :, 4],
                os.path.join(OUTPUT_DIR, 'ch_gt' + str(img_no) + '_' + str(4) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)

    # out_ens_gt = get_transformed_ens_gt2(ens_gt, augmentation_type, centered_transform, aug_transform,transformation_parameters_list)
    out_ens_gt = sitk.GetImageFromArray(ens_gt)
    out_ens_gt.SetSpacing(reference_spacing)
    out_ens_gt = augment_images_spatial(out_ens_gt, reference_image, augmentation_type, centered_transform,
                                        aug_transform, transformation_parameters_list, default_intensity_value=-1.0)
    out_ens_gt = sitk.GetArrayFromImage(out_ens_gt)
    out_ens_gt[:, :, :, 0:4] = np.where(out_ens_gt[:, :, :, 0:4] == -1, np.zeros_like(out_ens_gt[:, :, :, 0:4]),
                                        out_ens_gt[:, :, :, 0:4])
    out_ens_gt[:, :, :, 4] = np.where(out_ens_gt[:, :, :, 4] == -1, np.ones_like(out_ens_gt[:, :, :, 4]),
                                      out_ens_gt[:, :, :, 4])

    write_image(out_ens_gt[:, :, :, 0],
                os.path.join(OUTPUT_DIR, 'ch_ens_gt' + str(img_no) + '_' + str(0) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_ens_gt[:, :, :, 1],
                os.path.join(OUTPUT_DIR, 'ch_ens_gt' + str(img_no) + '_' + str(1) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_ens_gt[:, :, :, 2],
                os.path.join(OUTPUT_DIR, 'ch_ens_gt' + str(img_no) + '_' + str(2) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_ens_gt[:, :, :, 3],
                os.path.join(OUTPUT_DIR, 'ch_ens_gt' + str(img_no) + '_' + str(3) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)
    write_image(out_ens_gt[:, :, :, 4],
                os.path.join(OUTPUT_DIR, 'ch_ens_gt' + str(img_no) + '_' + str(4) + '_' + AugmentTypes(
                    augmentation_type).name + '.nrrd'), gt_ref)

    return out_img, out_gt, out_ens_gt

def get_single_image_augmentation(augmentation_type, orig_image, orig_gt):
    # print(img_no, augmentation_type)
    out_img = np.zeros([32, 168, 168, 1], dtype=np.float32)

    img1 = sitk.GetImageFromArray(orig_image[:, :, :, 0])
    img1.SetSpacing(reference_spacing)
    reference_image = get_reference_image(img1)

    img = sitk.GetImageFromArray(orig_image)
    img.SetSpacing(reference_spacing)
    centered_transform, aug_transform, transformation_parameters_list = get_augmentation_transform(img, reference_image,
                                                                                                   augmentation_type)

    # transform image
    res_img = augment_images_spatial(img, reference_image, augmentation_type, centered_transform,
                                     aug_transform, transformation_parameters_list)

    out_img[:, :, :, 0] = sitk.GetArrayFromImage(res_img)

    # transform gt
    gt_ref = sitk.GetImageFromArray(orig_gt)
    gt_ref.SetSpacing(reference_spacing)

    out_gt = get_transformed_gt(orig_gt, augmentation_type, centered_transform, aug_transform,
                                    transformation_parameters_list)


    return out_img, out_gt

if __name__ == '__main__':
    root_path = '/cache/suhita/data/prostate/fold_1_P1.0/train/'
    img_path = root_path + 'imgs/'
    # gt_path = '/home/suhita/zonals/temporal/sadv2/gt/'
    gt_path = root_path + 'gt/'
    ens_gt = '/data/suhita/temporal/prostate/output/sadv1/ens_gt/'
    img_no = 68
    img = np.load(img_path + str(img_no) + '.npy')
    gt = np.load(gt_path + str(img_no) + '.npy')
    ens_gt = np.load(ens_gt + str(img_no) + '.npy')
    flag = np.load('/data/suhita/temporal/prostate/output/sadv1/flag/' + str(img_no) + '.npy')
    augmentation_type = AugmentTypes.ROTATE.value

    out_img, out_gt, out_ens_gt, flag = get_single_image_augmentation_with_ensemble(augmentation_type, img, gt, ens_gt,
                                                                                    flag,
                                                                                    img_no, labelled_num=50)
    # out_img, out_gt = get_single_image_augmentation(augmentation_type, img, gt, img_no)
