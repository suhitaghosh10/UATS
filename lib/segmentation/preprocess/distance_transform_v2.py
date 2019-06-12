import SimpleITK as sitk
import numpy as np


def write_image(image, image_name, ref_image, is_image=False):
    if (True):
        writer = sitk.ImageFileWriter()
        writer.SetFileName(image_name)
        if not is_image:
            temp = sitk.GetImageFromArray(image)
        else:
            temp = image

        temp.CopyInformation(ref_image)
        writer.Execute(temp)


def generate_mask(gt_array_aniso):
    # img_array_aniso = np.load(img_path)

    no_imgs = gt_array_aniso.shape[0]
    reference_spacing = [0.5, 0.5, 3.0]
    gt_shape = [32, 168, 168, 5]
    zones_num = 5

    mask_arr = np.empty((no_imgs, *gt_shape))
    for gt_idx in np.arange(0, no_imgs):
        for zone in np.arange(0, zones_num):
            print('zone', zone, 'img', gt_idx)
            # OUTPUT_DIR = OUTPUT_DIR+ '/transform/'+str(gt_idx)+'_'+str(zone)+'/'
            # if not os.path.exists(out_dir+ '/transform/'+str(gt_idx)+'_'+str(zone)+'/'):
            #   os.mkdir(out_dir+ '/transform/'+str(gt_idx)+'_'+str(zone)+'/')

            gt_arr = gt_array_aniso[gt_idx, :, :, :, zone]
            gt = sitk.GetImageFromArray(gt_arr)
            gt.SetSpacing(reference_spacing)

            gt_dist = sitk.SignedMaurerDistanceMap(gt, insideIsPositive=True, squaredDistance=False,
                                                   useImageSpacing=True)
            temp = np.copy(sitk.GetArrayFromImage(gt_dist))
            # temp[temp <= -25] = 0
            # temp = np.abs(temp)
            max = np.reshape(np.max(temp, axis=(-2, -1)), (32, 1, 1))
            print('max shape', max.shape)
            min = np.reshape(np.min(temp, axis=(-2, -1)), (32, 1, 1))
            print('min shape', min.shape)
            temp = (temp - min) / (max - min)
            # print(temp)
            # if (gt_idx == 0):
            #   write_image(gt, os.path.join(OUTPUT_DIR, 'orig_gt' + str(zone) + '.nrrd'), gt, is_image=True)
            #    write_image(temp, os.path.join(OUTPUT_DIR, 'gt_dist' + str(zone) + '.nrrd'), gt, is_image=False)

            mask_arr[gt_idx, :, :, :, zone] = temp
    return mask_arr


if __name__ == '__main__':
    gt_path = '/home/suhita/zonals/data/training/gt/'
    OUTPUT_DIR = '/home/suhita/zonals/data/training/mask/'
    from lib.segmentation.utils import get_complete_array

    gt_arr = get_complete_array(gt_path, dtype='int8')
    arr = generate_mask(gt_arr)
    arr = arr.astype('float32')
    print(arr.shape)
    for idx in np.arange(arr.shape[0]):
        np.save(OUTPUT_DIR + str(idx), arr[idx])
        print(idx)
