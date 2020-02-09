import SimpleITK as sitk
import numpy as np
import os
from kits import utils
import csv

import math



BB_SIZE = [56, 152, 152]
SPACING = [4.0,1.0,1.0]


def getLabelStatistics():

    cases = sorted(os.listdir(files_dir))
    cases = [x for x in cases if 'case' in x]

    csv_file = open('kidney_segmentations.csv', 'w')
    csvWriter = csv.writer(csv_file, delimiter=';')
    csvWriter.writerow(['Case', 'kidney left', 'tumor left', 'kidney right', 'tumor right'])
    for i in range(210):

        segm = sitk.ReadImage(os.path.join(files_dir, cases[i], 'segmentation.nii.gz'))
        segm = utils.resampleImage(segm, [3.0, 0.75, 0.75], sitk.sitkNearestNeighbor, 0, sitk.sitkInt8)


        ### get bounding box
        size = segm.GetSize()
        size_x = size[0]
        size_y = size[1]
        size_z = size[2]

        region1 = sitk.RegionOfInterest(segm, [size_x, size_y, int(size_z / 2)], [0, 0, 0])
        region2 = sitk.RegionOfInterest(segm, [size_x, size_y, int(size_z / 2)], [0, 0, int(size_z / 2) - 1])
        # sitk.WriteImage(region1, 'region1.nrrd')
        # sitk.WriteImage(region2, 'region2.nrrd')
        filter = sitk.LabelShapeStatisticsImageFilter()

        ## left kidney
        filter.Execute(region1)
        nLabels = filter.GetNumberOfLabels()
        kidney_info = [cases[i]]
        if nLabels == 0:
            kidney_info = kidney_info + [0, 0]
        if nLabels == 1:
            kidney_info = kidney_info + [1, 0]
        if nLabels == 2:
            kidney_info = kidney_info + [1, 1]

        ## right kidney
        filter = sitk.LabelShapeStatisticsImageFilter()
        filter.Execute(region2)
        nLabels = filter.GetNumberOfLabels()
        if nLabels == 0:
            kidney_info = kidney_info + [0, 0]
        if nLabels == 1:
            kidney_info = kidney_info + [1, 0]
        if nLabels == 2:
            kidney_info = kidney_info + [1, 1]

        print(kidney_info)
        csvWriter.writerow(kidney_info)

    csv_file.close()

#
def getBoundingBoxes(img_left, img_right, segm):


    segm_left = utils.resampleToReference(segm, img_left, sitk.sitkNearestNeighbor, 0, out_dType=sitk.sitkUInt8)
    segm_right = utils.resampleToReference(segm, img_right, sitk.sitkNearestNeighbor, 0, out_dType=sitk.sitkUInt8)

    size = segm.GetSize()

    filter = sitk.LabelShapeStatisticsImageFilter()

    ## left kidney
    filter.Execute(segm_left)
    bb_left = filter.GetBoundingBox(1)


    ## right kidney
    filter = sitk.LabelShapeStatisticsImageFilter()
    filter.Execute(segm_right)
    bb_right = filter.GetBoundingBox(1)

    return bb_left, bb_right



def getSizes():

    cases = sorted(os.listdir(files_dir))
    cases = [x for x in cases if 'case' in x]

    for i in range(10):
        segm = sitk.ReadImage(os.path.join(files_dir, cases[i], 'segmentation.nii.gz'))
        segm = utils.resampleImage(segm, [3.0, 0.75, 0.75], sitk.sitkNearestNeighbor, 0, sitk.sitkInt8)

        print(cases[i], segm.GetSize())

# get bounding box
def getBB(img, label):

    filter = sitk.LabelShapeStatisticsImageFilter()

    filter.Execute(img)
    bb = filter.GetBoundingBox(label)

    return bb




def get_threshold_bounding_box(img, threshold_value=500):


    thresh= utils.binaryThresholdImage(img, threshold_value)
    ## todo: save thresholded!
    sitk.WriteImage(thresh, 'thresh_bones.nrrd')
    filter = sitk.LabelShapeStatisticsImageFilter()

    filter.Execute(thresh)
    nLabels = filter.GetNumberOfLabels()
    bb = filter.GetBoundingBox(1)

    #cropped = sitk.RegionOfInterest(img, [bb[3], bb[4], bb[5]], [bb[0], bb[1], bb[2]])
    # if bb[3]>100:
    #     sitk.WriteImage(cropped, 'cropped_'+str(i)+'.nrrd')

    # for lung extraction (similar intensities at the border would infer with lung extraction)
    smaller_bb = [bb[0], bb[1]+50, bb[2], bb[3], bb[4]-100 , bb[5] ]

    return bb, smaller_bb


def get_segmentation_statistics(segm, physical_centroid = True, return_BB = True):

    output = []

    filter = sitk.LabelShapeStatisticsImageFilter()


    filter.Execute(segm)
    nLabels = filter.GetNumberOfLabels()

    for i in range (1, nLabels+1):
        bb = filter.GetBoundingBox(i)
        size = filter.GetPhysicalSize(i)
        center = filter.GetCentroid(i)
        if not physical_centroid:
            center = segm.TransformPhysicalPointToIndex(center)

    if return_BB:
        return size, center, bb
    else:
        return size,center


def check_if_segm_in_ROI(cropped_img_right, cropped_img_left, segm):

    [size_x, size_y, size_z] = segm.GetSize()

    segm_left = sitk.RegionOfInterest(segm, [size_x, size_y, int(size_z / 2)], [0, 0, 0])
    segm_right = sitk.RegionOfInterest(segm, [size_x, size_y, int(size_z / 2)], [0, 0, int(size_z / 2) - 1])

    segm_right_cropped = utils.resampleToReference(segm_right, cropped_img_right, sitk.sitkNearestNeighbor, 0, out_dType=sitk.sitkUInt8)
    segm_left_cropped = utils.resampleToReference(segm_left, cropped_img_left, sitk.sitkNearestNeighbor, 0, out_dType=sitk.sitkUInt8)

    stats_right = get_segmentation_statistics(segm_right)[0]
    stats_right_cropped = get_segmentation_statistics(segm_right)[0]
    stats_left = get_segmentation_statistics(segm_left)[0]
    stats_left_cropped = get_segmentation_statistics(segm_left)[0]

    ratio_right = stats_right_cropped[0] / stats_right[0]
    ratio_left = stats_left_cropped[0] /stats_left[0]

    left_ok = False if abs(1-ratio_left) > 0.03 else True
    right_ok = False if abs(1 - ratio_right) > 0.03 else True

    print(left_ok, right_ok)

def get_left_and_right_BB():

    cases = sorted(os.listdir(files_dir))
    cases = [x for x in cases if 'case' in x]

    bbs_left = np.zeros([len(cases), 6])
    bbs_right = np.zeros([len(cases), 6])

    worried_cases = ['case_00048', 'case_00097', 'case_00178', 'case_00185', 'case_00191']

    for i in range(0, len(cases)):
        # if cases[i] in worried_cases:

        # if cases[i] == 'case_00001' or cases[i] == 'case_00011':
        print(cases[i])

        img = sitk.ReadImage(os.path.join(files_dir, cases[i], 'imaging.nii.gz'))
        img = utils.resampleImage(img, [4.0, 1.0, 1.0], sitk.sitkLinear, 0, sitk.sitkInt16)
        segm = sitk.ReadImage(os.path.join(files_dir, cases[i], 'segmentation.nii.gz'))

        # apply region of interest that is guided by bones
        bb, smaller_bb = get_threshold_bounding_box(img, 500)
        if bb[3] > 100:

            cropped_img = sitk.RegionOfInterest(img, [smaller_bb[3], smaller_bb[4], smaller_bb[5]],
                                                [smaller_bb[0], smaller_bb[1], smaller_bb[2]])
            # sitk.WriteImage(cropped_img, 'cropped_' + str(i) + '.nrrd')

            lung_img = sitk.BinaryThreshold(cropped_img, -900, -700, 1, 0)  # lung -900, -700
            #sitk.WriteImage(lung_img, 'lung_img_thresh' + str(i) + '.nrrd')

            lung_img = utils.getLargestConnectedComponents(lung_img)
            # sitk.WriteImage(lung_img, 'lung_img_cc' + str(i) + '.nrrd')

            # get boundingbox of lung:
            bb_lung = getBB(lung_img, 1)
            # print('bb_lung: ', bb_lung)

            # values for final ROI
            start_X = max(0, bb_lung[0] + bb_lung[3] - 20)  # start 20 voxel above bottom of lung
            size_X = 100
            if start_X + 100 > img.GetSize()[0]:
                start_X = img.GetSize()[0] - size_X - 1

            # img_final = sitk.RegionOfInterest(img, [size_X, bb[4], bb[5]],
            #                                   [start_X, bb[1], bb[2]])

            img_final = sitk.RegionOfInterest(img, [size_X, img.GetSize()[1], img.GetSize()[2]],
                                              [start_X,0,0])

            # sitk.WriteImage(cropped_img_final, 'final_cropped_' + str(i) + '.nrrd')

        else:
            ### pad volume in x direction

            # get minimum value for padding
            filter = sitk.StatisticsImageFilter()
            filter.Execute(img)
            minValue = filter.GetMinimum()

            img_final = utils.pad_volume(img, target_size_x=100, padValue=minValue)

        ## split image in left and right half
        [size_x, size_y, size_z] = img_final.GetSize()

        img_left = sitk.RegionOfInterest(img_final, [100, size_y, int(size_z / 2)], [0, 0, 0])
        img_right = sitk.RegionOfInterest(img_final, [100, size_y, int(size_z / 2)], [0, 0, int(size_z / 2) - 1])

        bb_left, bb_right = getBoundingBoxes(img_left, img_right, segm)
        bbs_left[i] = [bb_left[0], bb_left[1], bb_left[2], bb_left[3], bb_left[4], bb_left[5]]
        bbs_right[i] = [bb_right[0], bb_right[1], bb_right[2], bb_right[3] , bb_right[4] , bb_right[5] ]

        print(bb_left, bb_right)

    np.save('bbs_left.npy', bbs_left)
    np.save('bbs_right.npy', bbs_right)

    print('max and min x: ', np.amin(bbs_left[:, 0]), np.amax(bbs_left[:, 3]))
    print('max and min y: ', np.amin(bbs_left[:, 1]), np.amax(bbs_left[:, 4]))
    print('max and min z: ', np.amin(bbs_left[:, 2]), np.amax(bbs_left[:, 5]))

    print('max and min x: ', np.amin(bbs_right[:, 0]), np.amax(bbs_right[:, 3]))
    print('max and min y: ', np.amin(bbs_right[:, 1]), np.amax(bbs_right[:, 4]))
    print('max and min z: ', np.amin(bbs_right[:, 2]), np.amax(bbs_right[:, 5]))


def get_final_roi(img, start_x, start_y, start_z, size_x, size_y, size_z):

    img_size = img.GetSize()

    check_x = True if (start_x+size_x)<img_size[0] else False
    check_y = True if (start_y+size_y)<img_size[1] else False
    check_z = True if (start_z + size_z) < img_size[2] else False

    size = [int(size_x), int(size_y), int(size_z)]
    start = [int(start_x), int(start_y), int(start_z)]


    if check_x and check_y and check_z:
        out_img = sitk.RegionOfInterest(img, size , start)
    else:
        out_img = utils.pad_volume(img, target_size_x=start_x+size_x, target_size_y=start_y+size_y,
                                   target_size_z=start_z+size_z, padValue=utils.getMinimum(img))
        out_img = sitk.RegionOfInterest(out_img, size , start)

    return out_img

def remove_segmentations_in_background(img, segm):

    # get background
    thresholded = utils.binaryThresholdImage(img, -1023)
    out =  sitk.Multiply(thresholded, segm)
    return out


def normalizeIntensities(*imgs):
    out = []

    for img in imgs:
        array = np.ndarray.flatten(sitk.GetArrayFromImage(img))

        upperPerc = np.percentile(array, 99)  # 98
        lowerPerc = np.percentile(array, 1)  # 2
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


def preprocess():

    utils.makedir(preprocessed_dir)

    bb_size = [80, 200, 200] # will be extended by [10,20,20] per side for augmentation

    # get_left_and_right_BB()
    cases = sorted(os.listdir(files_dir))
    cases = [x for x in cases if 'case' in x]

    bbs_left = np.load('bbs_left.npy')
    bbs_right = np.load('bbs_right.npy')

    for i in range(0, len(cases)):
        #if cases[i] == 'case_00002':
        out_dir = os.path.join(preprocessed_dir,cases[i])
        utils.makedir(out_dir)

        print(cases[i])

        img = sitk.ReadImage(os.path.join(files_dir, cases[i], 'imaging.nii.gz'))
        img = utils.resampleImage(img, [3.0, 0.75, 0.75], sitk.sitkLinear, 0, sitk.sitkInt16)
        segm = sitk.ReadImage(os.path.join(files_dir, cases[i], 'segmentation.nii.gz'))

        # apply region of interest that is guided by bones
        bb, smaller_bb = get_threshold_bounding_box(img, 500)
        if bb[3] > 100:

            cropped_img = sitk.RegionOfInterest(img, [smaller_bb[3], smaller_bb[4], smaller_bb[5]],
                                                [smaller_bb[0], smaller_bb[1], smaller_bb[2]])
            # sitk.WriteImage(cropped_img, 'cropped_' + str(i) + '.nrrd')

            lung_img = sitk.BinaryThreshold(cropped_img, -900, -700, 1, 0)  # lung -900, -700
            #sitk.WriteImage(lung_img, 'lung_img_thresh' + str(i) + '.nrrd')

            lung_img = utils.getLargestConnectedComponents(lung_img)
           # sitk.WriteImage(lung_img, 'lung_img_cc' + str(i) + '.nrrd')

            # get boundingbox of lung:
            bb_lung = getBB(lung_img, 1)
            # print('bb_lung: ', bb_lung)

            # values for final ROI
            start_X = max(0, bb_lung[0] + bb_lung[3] - 20)  # start 20 voxel above bottom of lung
            size_X = 100
            if start_X + 100 > img.GetSize()[0]:
                start_X = img.GetSize()[0] - size_X - 1

            img = sitk.RegionOfInterest(img, [size_X, img.GetSize()[1], img.GetSize()[2]],
                                              [start_X, 0, 0])
            # img = sitk.RegionOfInterest(img, [size_X, bb[4], bb[5]],
            #                                   [start_X, bb[1], bb[2]])

            sitk.WriteImage(img, 'final_cropped_' + str(i) + '.nrrd')

        else:
            ### pad volume in x direction
            img = utils.pad_volume(img, target_size_x=100, padValue=utils.getMinimum(img))

        ## split image in left and right half
        [size_x, size_y, size_z] = img.GetSize()

        img_left = sitk.RegionOfInterest(img, [size_x, size_y, int(size_z / 2)], [0, 0, 0])
        img_right = sitk.RegionOfInterest(img, [size_x, size_y, int(size_z / 2)], [0, 0, int(size_z / 2) - 1])

        sitk.WriteImage(img_left, 'img_left_temp.nrrd')
        sitk.WriteImage(img_right, 'img_right_temp.nrrd')

       # get ROI for left and right kidney
        roi_left_start = [max(0,bbs_left[i,0]-10), max(0,bbs_left[i,1]-10), max(0,bbs_left[i,2]-10)]
        roi_right_start =  [max(0,bbs_right[i,0]-10), max(0,bbs_right[i,1]-10), max(0,bbs_right[i,2]-10)]

        # apply ROI to image
        img_left_final = get_final_roi(img_left, roi_left_start[0], roi_left_start[1], roi_left_start[2],
                      bb_size[0]+20, bb_size[1]+20, bb_size[2]+20)
        img_right_final = get_final_roi(img_right, roi_right_start[0], roi_right_start[1], roi_right_start[2],
                      bb_size[0]+20, bb_size[1]+20, bb_size[2]+20)

        segm_left_final = utils.resampleToReference(segm, img_left_final, sitk.sitkNearestNeighbor,
                                                    defaultValue=0, out_dType=sitk.sitkUInt8)

        segm_right_final = utils.resampleToReference(segm, img_right_final, sitk.sitkNearestNeighbor,
                                                    defaultValue=0, out_dType=sitk.sitkUInt8)


        # write images
        sitk.WriteImage(img_left_final, os.path.join(out_dir,'img_left.nrrd'))
        sitk.WriteImage(img_right_final, os.path.join(out_dir, 'img_right.nrrd'))
        sitk.WriteImage(segm_left_final, os.path.join(out_dir, 'segm_left.nrrd'))
        sitk.WriteImage(segm_right_final, os.path.join(out_dir, 'segm_right.nrrd'))

def preprocess_centered_BB():

    utils.makedir(preprocessed_dir)


    # get_left_and_right_BB()
    cases = sorted(os.listdir(files_dir))
    cases = [x for x in cases if 'case' in x]

    # bbs_left = np.load('bbs_left.npy')
    # bbs_right = np.load('bbs_right.npy')

    for i in range(0, len(cases)):
        # if cases[i] == 'case_00002':
        out_dir = os.path.join(preprocessed_dir, cases[i])
        utils.makedir(out_dir)

        print(cases[i])

        img = sitk.ReadImage(os.path.join(files_dir, cases[i], 'imaging.nii.gz'))
        img = utils.resampleImage(img, SPACING, sitk.sitkLinear, 0, sitk.sitkInt16)
        segm = sitk.ReadImage(os.path.join(files_dir, cases[i], 'segmentation.nii.gz'))
        # threhsold segmentation such that no tumor tissue is available anymore
        segm = sitk.BinaryThreshold(segm, 1, 2, 1, 0)

         ## split image in left and right half
        [size_x, size_y, size_z] = img.GetSize()

        img_left = sitk.RegionOfInterest(img, [size_x, size_y, int(size_z / 2)], [0, 0, 0])
        img_right = sitk.RegionOfInterest(img, [size_x, size_y, int(size_z / 2)], [0, 0, int(size_z / 2) - 1])

        img_left, img_right = normalizeIntensities(img_left, img_right)

        segm_left = utils.resampleToReference(segm, img_left, sitk.sitkNearestNeighbor,
                                                    defaultValue=0, out_dType=sitk.sitkUInt8)

        segm_right = utils.resampleToReference(segm, img_right, sitk.sitkNearestNeighbor,
                                                     defaultValue=0, out_dType=sitk.sitkUInt8)

        sitk.WriteImage(segm_left, 'segm_left.nrrd')
        sitk.WriteImage(segm_right, 'segm_right.nrrd')

        size_l, centroid_l = get_segmentation_statistics(segm_left, False, False)
        size_r, centroid_r = get_segmentation_statistics(segm_right, False, False)

        start_l_x = max(0,centroid_l[0] - int(BB_SIZE[0] / 2))
        start_l_y = max(0, centroid_l[1] - int(BB_SIZE[1] / 2))
        start_l_z = max(0, centroid_l[2] - int(BB_SIZE[2] / 2))

        start_r_x = max(0, centroid_r[0] - int(BB_SIZE[0] / 2))
        start_r_y = max(0, centroid_r[1] - int(BB_SIZE[1] / 2))
        start_r_z = max(0, centroid_r[2] - int(BB_SIZE[2] / 2))

        segm_l_final = get_final_roi(segm_left, start_l_x, start_l_y, start_l_z,
                                     BB_SIZE[0], BB_SIZE[1], BB_SIZE[2])
        segm_r_final = get_final_roi(segm_right, start_r_x, start_r_y, start_r_z,
                                     BB_SIZE[0], BB_SIZE[1], BB_SIZE[2])

        img_l_final = get_final_roi(img_left, start_l_x, start_l_y, start_l_z,
                                     BB_SIZE[0], BB_SIZE[1], BB_SIZE[2])
        img_r_final = get_final_roi(img_right, start_r_x, start_r_y, start_r_z,
                                     BB_SIZE[0], BB_SIZE[1], BB_SIZE[2])


        print(size_l, size_r)

        # write images
        sitk.WriteImage(img_l_final, os.path.join(out_dir, 'img_left.nrrd'))
        sitk.WriteImage(img_r_final, os.path.join(out_dir, 'img_right.nrrd'))
        sitk.WriteImage(segm_l_final, os.path.join(out_dir, 'segm_left.nrrd'))
        sitk.WriteImage(segm_r_final, os.path.join(out_dir, 'segm_right.nrrd'))

def preprocess_unlabeled(files_dir, preprocessed_dir):
    utils.makedir(preprocessed_dir)

    #bb_size = [80, 200, 200]  # will be extended by [10,20,20] per side for augmentation

    # get_left_and_right_BB()
    cases = sorted(os.listdir(files_dir))
    cases = [x for x in cases if 'case' in x]


    for i in range(0, len(cases)):
       # if cases[i] == 'case_00223':
        out_dir = os.path.join(preprocessed_dir, cases[i])
        utils.makedir(out_dir)

        print(cases[i])

        img = sitk.ReadImage(os.path.join(files_dir, cases[i], 'imaging.nii.gz'))
        img = utils.resampleImage(img, SPACING, sitk.sitkLinear, 0, sitk.sitkInt16)


        # apply region of interest that is guided by bones
        bb, smaller_bb = get_threshold_bounding_box(img, 500)



        if img.GetSize()[0] > BB_SIZE[0]:

            #if img.GetSize()[0] > BB_SIZE[1]+50: # if x-axis is very long, it might be a whole body scan

            cropped_img = sitk.RegionOfInterest(img, [smaller_bb[3], smaller_bb[4], smaller_bb[5]],
                                                [smaller_bb[0], smaller_bb[1], smaller_bb[2]]
                                                )
            sitk.WriteImage(cropped_img, 'cropped_' + str(i) + '.nrrd')

            lung_img = sitk.BinaryThreshold(cropped_img, -900, -700, 1, 0)  # lung -900, -700
            sitk.WriteImage(lung_img, 'lung_img_thresh' + str(i) + '.nrrd')

            lung_img = utils.getLargestConnectedComponents(lung_img)
            sitk.WriteImage(lung_img, 'lung_img_cc' + str(i) + '.nrrd')

            # get boundingbox of lung:
            bb_lung = getBB(lung_img, 1)
            start_y_w_bb_lung = cropped_img.TransformIndexToPhysicalPoint([bb_lung[0], bb_lung[1], bb_lung[2]])
            start_y_w_img = img.TransformPhysicalPointToIndex(start_y_w_bb_lung)

            # print('bb_lung: ', bb_lung)

            # values for final ROI
            start_X = max(0, bb_lung[0] + bb_lung[3]) #- 0.1*BB_SIZE[0])  # start 20 voxel above bottom of lung
            size_X = BB_SIZE[0]
            if start_X + BB_SIZE[0] > img.GetSize()[0]:
                start_X = img.GetSize()[0] - size_X - 1

            start_y = max(0,(start_y_w_img[1]+bb_lung[4]-BB_SIZE[1])) ## according to bones bounding box
            #start_y = max(0, bb[1] + bb[4] - BB_SIZE[1])
            start_z_r = int((bb[2] + bb[5]) / 2)
            start_z_l = max(0, int((bb[2] + bb[5]) / 2) - BB_SIZE[2])


        else:
            ### pad volume in x direction
            img = utils.pad_volume(img, target_size_x=BB_SIZE[0], padValue=utils.getMinimum(img))

            # values for final ROI
            start_X = 0  # start 20 voxel above bottom of lung
            start_y = max(0, (bb[1] + bb[4]+20 - BB_SIZE[1]))  ## according to bones bounding box
            start_z_r = int((bb[2] + bb[5]) / 2)
            start_z_l = max(0,int((bb[2] + bb[5]) / 2)-BB_SIZE[2])


        left_img = get_final_roi(img, start_X, start_y, start_z_l,
                                 BB_SIZE[0], BB_SIZE[1], BB_SIZE[2])
        right_img = get_final_roi(img, start_X, start_y, start_z_r,
                                  BB_SIZE[0], BB_SIZE[1], BB_SIZE[2])

        sitk.WriteImage(left_img, os.path.join(out_dir, 'img_left.nrrd'))
        sitk.WriteImage(right_img, os.path.join(out_dir, 'img_right.nrrd'))


if __name__ == '__main__':

    files_dir = '/data/anneke/kits_challenge/kits19/data/unlabelled'
    preprocessed_dir = '/data/anneke/kits_challenge/kits19/data/preprocessed_unlabeled'

    #################################
    # get Bounding box positions and sizes
    #get_left_and_right_BB()


    ##################################
    ### actual preprocessing ###
    #preprocess_centered_BB()

    preprocess_unlabeled(files_dir = '/data/anneke/kits_challenge/kits19/data/unlabelled',
                         preprocessed_dir = '/data/anneke/kits_challenge/kits19/data/preprocessed_unlabeled')




########################################
#remove segmentations from other kidney in the background

    # cases = sorted(os.listdir(files_dir))
    # for i in range(0, len(cases)):
    #     #if cases[i] == 'case_00002':
    #     out_dir = os.path.join(preprocessed_dir,cases[i])
    #
    #     print(cases[i])
    #
    #     img_l = sitk.ReadImage(os.path.join(out_dir, 'img_left.nrrd'))
    #     img_r = sitk.ReadImage(os.path.join(out_dir, 'img_right.nrrd'))
    #     segm_l = sitk.ReadImage(os.path.join(out_dir, 'segm_left.nrrd'))
    #     segm_r = sitk.ReadImage(os.path.join(out_dir, 'segm_right.nrrd'))
    #
    #     segm_left_final = remove_segmentations_in_background(img_l, segm_l)
    #     segm_right_final = remove_segmentations_in_background(img_r, segm_r)
    #
    #     sitk.WriteImage(segm_left_final, os.path.join(out_dir, 'segm_left.nrrd'))
    #     sitk.WriteImage(segm_right_final, os.path.join(out_dir, 'segm_right.nrrd'))


########################################
   # save as npy arrays

    # cases = sorted(os.listdir(preprocessed_dir))
    # for i in range(0, len(cases)):
    #     #if cases[i] == 'case_00002':
    #     out_dir = os.path.join(preprocessed_dir,cases[i])
    #
    #     print(cases[i])
    #
    #     img_l = sitk.ReadImage(os.path.join(out_dir, 'img_left.nrrd'))
    #     img_r = sitk.ReadImage(os.path.join(out_dir, 'img_right.nrrd'))
    #     # segm_l = sitk.ReadImage(os.path.join(out_dir, 'segm_left.nrrd'))
    #     # segm_r = sitk.ReadImage(os.path.join(out_dir, 'segm_right.nrrd'))
    #
    #     np.save(os.path.join(out_dir, 'img_left.npy'), sitk.GetArrayFromImage(img_l))
    #     np.save(os.path.join(out_dir, 'img_right.npy'), sitk.GetArrayFromImage(img_r))
    #     # np.save(os.path.join(out_dir, 'segm_left.npy'), sitk.GetArrayFromImage(segm_l))
    #     # np.save(os.path.join(out_dir, 'segm_right.npy'), sitk.GetArrayFromImage(segm_r))
    #
    # ######other stuff
    #
    # # save median slice of training cases to jpeg
    #
    # cases = sorted(os.listdir(preprocessed_dir))
    # for i in range(0, len(cases)):
    #     #if cases[i] == 'case_00002':
    #     out_dir = os.path.join('screenshots')
    #
    #     print(cases[i])
    #     img_l =np.load(os.path.join(preprocessed_dir, cases[i], 'img_left.npy'))
    #     img_r = np.load(os.path.join(preprocessed_dir, cases[i], 'img_right.npy'))
    #     # segm_l = np.load(os.path.join(preprocessed_dir, cases[i], 'segm_left.npy'))
    #     # segm_r = np.load(os.path.join(preprocessed_dir, cases[i], 'segm_right.npy'))
    #
    #     cv2.imwrite(os.path.join(out_dir, cases[i]+ '_img_left.jpeg'), img_l[:,:,25])
    #     cv2.imwrite(os.path.join(out_dir, cases[i] + '_img_right.jpeg'),  img_r[:,:,25])
    #     # cv2.imwrite(os.path.join(out_dir, cases[i] + '_segm_left.jpeg'),  segm_l[:,:,25]*255)
    #     # cv2.imwrite(os.path.join(out_dir, cases[i] + '_segm_right.jpeg'), segm_r[:, :, 25]*255)



    #############


    # cases = sorted(os.listdir(files_dir))
    # cases = [x for x in cases if 'case' in x]
    #
    # bbs_left = np.load('bbs_left.npy')
    # bbs_right = np.load('bbs_right.npy')
    #
    # for i in range(0, len(cases)):
    #
    #     print( cases[i], bbs_right[i])
    # #
    # print('max and min x: ', np.amin(bbs_left[:, 0]), np.amax(bbs_left[:, 3]))
    # print('max and min y: ', np.amin(bbs_left[:, 1]), np.amax(bbs_left[:, 4]))
    # print('max and min z: ', np.amin(bbs_left[:, 2]), np.amax(bbs_left[:, 5]))
    #
    # print('max and min x: ', np.amin(bbs_right[:, 0]), np.amax(bbs_right[:, 3]))
    # print('max and min y: ', np.amin(bbs_right[:, 1]), np.amax(bbs_right[:, 4]))
    # print('max and min z: ', np.amin(bbs_right[:, 2]), np.amax(bbs_right[:, 5]))






        # sitk.WriteImage(img_final, cases[i] + '_final.nrrd')
        # sitk.WriteImage(img_left,cases[i] + '_crop_left.nrrd')
        # sitk.WriteImage(img_right, cases[i] + '_crop_right.nrrd')

        # check if cropped image contains segmentation:
        #check_if_segm_in_ROI(img_right, img_left, segm)








            # see if values need to

            # sizeX_cropped_img = bb[3]
            #
            # cropFilter = sitk.CropImageFilter()
            # cropFilter.SetLowerBoundaryCropSize([start_X,0,0])
            # x_crop_upper = sizeX_cropped_img - (start_X+bb_lung[3])
            # cropFilter.SetUpperBoundaryCropSize([x_crop_upper, 0, 0])
            # x_cropped_img = cropFilter.Execute(cropped_img_temp)








        # get lung
        #


    #getBonesBoundingBox()
    #getSizes()


