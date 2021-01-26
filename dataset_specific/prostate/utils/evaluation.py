import csv

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from dataset_specific.prostate.utils.preprocess import *


def getDice(prediction, groundTruth):
    filter = sitk.LabelOverlapMeasuresImageFilter()
    filter.Execute(prediction, groundTruth)
    dice = filter.GetDiceCoefficient()
    return dice


def relativeAbsoluteVolumeDifference(prediction, groundTruth):
    # get number of pixels in segmentation
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedComponents = connectedFilter.Execute(prediction)
    labelFilter = sitk.LabelShapeStatisticsImageFilter()
    labelFilter.Execute(connectedComponents)
    x = labelFilter.GetNumberOfLabels()
    if x == 0:
        return 0.0
    pixelsPrediction = labelFilter.GetNumberOfPixels(1)

    connectedComponents = connectedFilter.Execute(groundTruth)
    labelFilter = sitk.LabelShapeStatisticsImageFilter()
    labelFilter.Execute(connectedComponents)
    pixelsGT = labelFilter.GetNumberOfPixels(1)

    # compute and return relative absolute Volume Difference
    return (abs((pixelsPrediction / pixelsGT) - 1)) * 100


def getBoundaryDistances(prediction, groundTruth):
    # get surfaces
    contourP = sitk.BinaryContour(prediction, fullyConnected=True)
    maxFilter = sitk.MinimumMaximumImageFilter()
    maxFilter.Execute(contourP)
    x = maxFilter.GetMaximum()
    if maxFilter.GetMaximum() == 0:
        return (0.0, 0.0)
    contourGT = sitk.BinaryContour(groundTruth, fullyConnected=True)

    contourP_dist = sitk.DanielssonDistanceMap(contourP, inputIsBinary=True, squaredDistance=False,
                                               useImageSpacing=True)
    contourGT_dist = sitk.DanielssonDistanceMap(contourGT, inputIsBinary=True, squaredDistance=False,
                                                useImageSpacing=True)

    # image with directed distance from prediction contour to GT contour
    contourP_masked = sitk.Mask(contourP_dist, contourGT)

    # image with directed distance from GT contour to predicted contour
    contourGT_masked = sitk.Mask(contourGT_dist, contourP)
    # sitk.WriteImage(contourGT_masked, 'contourGT_masked.nrrd')
    sitk.WriteImage(contourP_masked, 'contourP_masked.nrrd')

    contourP_arr = sitk.GetArrayFromImage(contourP)
    contourP_arr = contourP_arr.astype(np.bool)
    contourP_arr_inv = np.invert(contourP_arr)
    contourGT_arr = sitk.GetArrayFromImage(contourGT)
    contourGT_arr = contourGT_arr.astype(np.bool)
    contourGT_arr_inv = np.invert(contourGT_arr)

    dist_PredtoGT = sitk.GetArrayFromImage(contourP_masked)
    dist_PredtoGT = np.ma.masked_array(dist_PredtoGT, contourGT_arr_inv).compressed()

    dist_GTtoPred = sitk.GetArrayFromImage(contourGT_masked)
    dist_GTtoPred = np.ma.masked_array(dist_GTtoPred, contourP_arr_inv).compressed()

    hausdorff = max(np.percentile(dist_PredtoGT, 95), np.percentile(dist_GTtoPred, 95))
    distances = np.concatenate((dist_PredtoGT, dist_GTtoPred))
    mean = distances.mean()
    return (hausdorff, mean)


def evaluateFiles_zones(GT_array, prediction_arr, csvName):
    with open(csvName + '.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';', lineterminator='\n',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(
            ['Case', 'PZ Dice', 'CZ Dice', 'US Dice', 'AFS Dice', 'Bg Dice', 'PZ MeanDis', 'CZ MeanDis', 'US MeanDis',
             'AFS MeanDis', 'Bg MeanDis'])

        nrImgs = GT_array.shape[0]
        dices = np.zeros((nrImgs, 5), dtype=np.float32)
        print(dices.shape)
        mad = np.zeros((nrImgs, 5), dtype=np.float32)

        for imgNumber in range(0, nrImgs):
            print('Case' + str(imgNumber))
            values = ['Case' + str(imgNumber)]
            temp_dice = []
            temp_mad = []
           # predictions = np.load(prediction_arr + '.npy')
            for zoneIndex in range(0, 5):
                pred_arr = prediction_arr[imgNumber, :, :, :, zoneIndex]
                pred_arr = thresholdArray(pred_arr, 0.5)
                # pred_arr = pred_arr.astype(int)
                # maxValue = np.max(pred_arr)
                pred_img = sitk.GetImageFromArray(pred_arr)

                GT_label = sitk.GetImageFromArray(GT_array[imgNumber, :, :, :, zoneIndex])
                #####pred_img = utils.resampleToReference(pred_img, GT_label, sitk.sitkNearestNeighbor, 0)
                pred_img = castImage(pred_img, sitk.sitkUInt8)
                ####GT_label = utils.resampleToReference(GT_label, pred_img, sitk.sitkNearestNeighbor, 0)
                GT_label = castImage(GT_label, sitk.sitkUInt8)

                # sitk.WriteImage(pred_img, 'predImg.nrrd')
                # sitk.WriteImage(GT_label, 'GT_label.nrrd')

                dice = getDice(pred_img, GT_label)
                temp_dice.append(dice)
                print(dice)
                # avd = relativeAbsoluteVolumeDifference(pred_img, GT_label)
                [hausdorff, avgDist] = getBoundaryDistances(pred_img, GT_label)
                temp_mad.append(avgDist)
                # values.append(dice)
                # values.append(avgDist)
                dices[imgNumber, zoneIndex] = dice
                mad[imgNumber, zoneIndex] = avgDist

            values.append(temp_dice[0])
            values.append(temp_dice[1])
            values.append(temp_dice[2])
            values.append(temp_dice[3])
            values.append(temp_dice[4])

            values.append(temp_mad[0])
            values.append(temp_mad[1])
            values.append(temp_mad[2])
            values.append(temp_mad[3])
            values.append(temp_mad[4])
            # values.append(temp_mad)
            csvwriter.writerow(values)

        csvwriter.writerow('')
        average = ['Average', np.average(dices[:, 0]), np.average(dices[:, 1]), np.average(dices[:, 2]),
                   np.average(dices[:, 3]), np.average(dices[:, 4]), np.average(mad[:, 0]), np.average(mad[:, 1]),
                   np.average(mad[:, 2]),
                   np.average(mad[:, 3]), np.average(mad[:, 4])]
        median = ['Median', np.median(dices[:, 0]), np.median(dices[:, 1]), np.median(dices[:, 2]),
                  np.median(dices[:, 3]), np.median(dices[:, 4]), np.median(mad[:, 0]), np.median(mad[:, 1]),
                  np.median(mad[:, 2]),
                  np.median(mad[:, 3]), np.median(dices[:, 3])]
        std = ['STD', np.std(dices[:, 0]), np.std(dices[:, 1]), np.std(dices[:, 2]),
               np.std(dices[:, 3]), np.std(dices[:, 4]), np.std(mad[:, 0]), np.std(mad[:, 1]), np.std(mad[:, 2]),
               np.std(mad[:, 3]), np.std(dices[:, 4])]

        csvwriter.writerow(average)
        csvwriter.writerow(median)
        csvwriter.writerow(std)

        print('Dices')
        print(np.average(dices[:, 0]))
        print(np.average(dices[:, 1]))
        print(np.average(dices[:, 2]))
        print(np.average(dices[:, 3]))
        print(np.average(dices[:, 4]))

        print('Mean Dist')
        print(np.average(mad[:, 0]))
        print(np.average(mad[:, 1]))
        print(np.average(mad[:, 2]))
        print(np.average(mad[:, 3]))
        print(np.average(mad[:, 4]))


def evaluateFiles(GT_directory, pred_directory, csvName):
    with open(csvName + '.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';', lineterminator='\n',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['Case', 'Dice', 'Average Volume Difference', '95-Hausdorff', 'Avg Hausdorff'])

        cases = os.listdir(GT_directory)

        for case in cases:
            pred_img = sitk.ReadImage(pred_directory + case + '/predicted_CC.nrrd')
            print(case[:-1])
            GT_label = sitk.ReadImage(GT_directory + case + '/Segmentation-label_whole.nrrd')
            GT_label = castImage(GT_label, sitk.sitkUInt8)
            pred_img = resampleToReference(pred_img, GT_label, sitk.sitkNearestNeighbor, 0)
            pred_img = castImage(pred_img, sitk.sitkUInt8)
            dice = getDice(pred_img, GT_label)
            print(dice)
            avd = relativeAbsoluteVolumeDifference(pred_img, GT_label)
            [hausdorff, avgDist] = getBoundaryDistances(pred_img, GT_label)

            csvwriter.writerow([case, dice, avd, hausdorff, avgDist])


def visualizeResults(directory, img_tra, img_cor, img_sag, pred_img, GT_img, i):
    if not os.path.exists(directory + 'visualResults'):
        os.makedirs(directory + 'visualResults')

    contour = sitk.BinaryContour(pred_img, True)
    colors = [(1, 0, 0), (0, 0, 0)]  # R -> G -> B
    n_bins = 5  # Discretizes the interpolation into bins
    cmap_name = 'my_list'

    contourGT = sitk.BinaryContour(GT_img, True)
    colorsGT = [(1, 1, 0), (0, 0, 0)]  # R -> G -> B
    cmap_nameGT = 'GT'

    contourArray = sitk.GetArrayFromImage(contour)
    contourArray = np.flip(contourArray, axis=0)
    contourArrayGT = sitk.GetArrayFromImage(contourGT)
    contourArrayGT = np.flip(contourArrayGT, axis=0)
    imgArray_tra = sitk.GetArrayFromImage(img_tra)
    imgArray_tra = np.flip(imgArray_tra, axis=0)
    imgArray_cor = sitk.GetArrayFromImage(img_cor)
    imgArray_cor = np.flip(imgArray_cor, axis=0)
    imgArray_sag = sitk.GetArrayFromImage(img_sag)
    imgArray_sag = np.flip(imgArray_sag, axis=0)

    masked_data = np.ma.masked_where(contourArray == 0, contourArray)
    masked_dataGT = np.ma.masked_where(contourArrayGT == 0, contourArrayGT)
    z = imgArray_tra.shape[0]

    plt.subplots(3, 3, figsize=(30, 30))

    colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    colormapGT = LinearSegmentedColormap.from_list(cmap_nameGT, colorsGT, N=n_bins)
    plt.subplot(3, 3, 1)
    plt.imshow(imgArray_tra[int(z / 3), :, :], 'gray')
    plt.imshow(masked_data[int(z / 3), :, :], cmap=colormap, interpolation='none')
    plt.imshow(masked_dataGT[int(z / 3), :, :], cmap=colormapGT, interpolation='none')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(imgArray_tra[int(z / 2), :, :], 'gray')
    plt.imshow(masked_data[int(z / 2), :, :], cmap=colormap, interpolation='none')
    plt.imshow(masked_dataGT[int(z / 2), :, :], cmap=colormapGT, interpolation='none')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(imgArray_tra[int(2 * (z / 3)), :, :], 'gray')
    plt.imshow(masked_data[int(2 * (z / 3)), :, :], cmap=colormap, interpolation='none')
    plt.imshow(masked_dataGT[int(2 * (z / 3)), :, :], cmap=colormapGT, interpolation='none')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(imgArray_cor[:, int(z / 3), :], 'gray')
    plt.imshow(masked_data[:, int(z / 3), :], cmap=colormap, interpolation='none')
    plt.imshow(masked_dataGT[:, int(z / 3), :], cmap=colormapGT, interpolation='none')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(imgArray_cor[:, int(z / 2), :], 'gray')
    plt.imshow(masked_data[:, int(z / 2), :], cmap=colormap, interpolation='none')
    plt.imshow(masked_dataGT[:, int(z / 2), :], cmap=colormapGT, interpolation='none')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(imgArray_cor[:, int(2 * (z / 3)), :], 'gray')
    plt.imshow(masked_data[:, int(2 * (z / 3)), :], cmap=colormap, interpolation='none')
    plt.imshow(masked_dataGT[:, int(2 * (z / 3)), :], cmap=colormapGT, interpolation='none')
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.imshow(imgArray_sag[:, :, int(z / 3)], 'gray')
    plt.imshow(masked_data[:, :, int(z / 3)], cmap=colormap, interpolation='none')
    plt.imshow(masked_dataGT[:, :, int(z / 3)], cmap=colormapGT, interpolation='none')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(imgArray_sag[:, :, int(z / 2)], 'gray')
    plt.imshow(masked_data[:, :, int(z / 2)], cmap=colormap, interpolation='none')
    plt.imshow(masked_dataGT[:, :, int(z / 2)], cmap=colormapGT, interpolation='none')
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.imshow(imgArray_sag[:, :, int(2 * (z / 3))], 'gray')
    plt.imshow(masked_data[:, :, int(2 * (z / 3))], cmap=colormap, interpolation='none')
    plt.imshow(masked_dataGT[:, :, int(z / 3)], cmap=colormapGT, interpolation='none')
    plt.axis('off')

    plt.savefig(directory + 'visualResults/img_' + str(i) + '.png')


def visualizeResultsSmall(directory, img_tra, img_cor, img_sag, pred_img, GT_img, i):
    if not os.path.exists(directory + 'visualResults'):
        os.makedirs(directory + 'visualResults')

    contour = sitk.BinaryContour(pred_img, True)
    colors = [(1, 0, 0), (0, 0, 0)]  # R -> G -> B
    n_bins = 5  # Discretizes the interpolation into bins
    cmap_name = 'my_list'

    contourGT = sitk.BinaryContour(GT_img, True)
    colorsGT = [(1, 1, 0), (0, 0, 0)]  # R -> G -> B
    cmap_nameGT = 'GT'

    contourArray = sitk.GetArrayFromImage(contour)
    contourArray = np.flip(contourArray, axis=0)
    contourArrayGT = sitk.GetArrayFromImage(contourGT)
    contourArrayGT = np.flip(contourArrayGT, axis=0)
    imgArray_tra = sitk.GetArrayFromImage(img_tra)
    imgArray_tra = np.flip(imgArray_tra, axis=0)
    imgArray_cor = sitk.GetArrayFromImage(img_cor)
    imgArray_cor = np.flip(imgArray_cor, axis=0)
    imgArray_sag = sitk.GetArrayFromImage(img_sag)
    imgArray_sag = np.flip(imgArray_sag, axis=0)

    masked_data = np.ma.masked_where(contourArray == 0, contourArray)
    masked_dataGT = np.ma.masked_where(contourArrayGT == 0, contourArrayGT)
    z = imgArray_tra.shape[0]

    plt.subplots(3, 3, figsize=(30, 30))

    colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    colormapGT = LinearSegmentedColormap.from_list(cmap_nameGT, colorsGT, N=n_bins)

    plt.subplot(1, 3, 1)
    plt.imshow(imgArray_tra[int(z / 2), :, :], 'gray')
    plt.imshow(masked_data[int(z / 2), :, :], cmap=colormap, interpolation='none')
    plt.imshow(masked_dataGT[int(z / 2), :, :], cmap=colormapGT, interpolation='none')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(imgArray_sag[:, :, int(z / 2)], 'gray')
    plt.imshow(masked_data[:, :, int(z / 2)], cmap=colormap, interpolation='none')
    plt.imshow(masked_dataGT[:, :, int(z / 2)], cmap=colormapGT, interpolation='none')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(imgArray_cor[:, int(2 * (z / 3)), :], 'gray')
    plt.imshow(masked_data[:, int(2 * (z / 3)), :], cmap=colormap, interpolation='none')
    plt.imshow(masked_dataGT[:, int(2 * (z / 3)), :], cmap=colormapGT, interpolation='none')
    plt.axis('off')

    plt.savefig(directory + 'visualResults/img_' + str(i) + '.png')


def regionBasedEvaluation(directory, csvName):
    with open(csvName + '.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';', lineterminator='\n',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(
            ['Case', 'Dice Apex', 'Dice Mid', 'Dice Base', 'AVD Apex', 'AVD Mid', 'AVD Base', '95-Hausdorff Apex',
             '95-Hausdorff Mid', '95-Hausdorff Base', 'Avg Dis Apex', 'Avg Dis Mid', 'Avg Dis Base'])

        for i in range(0, 15):
            predicted_array = np.load(directory + '/predicted_test_' + str(i) + '.npy')
            GT_array = np.load(directory + 'imgs_test_GT_' + str(i) + '.npy')
            GT_array = GT_array.astype(np.uint8)
            pred_img = sitk.GetImageFromArray(predicted_array[0, :, :, :, 0])

            pred_img = binaryThresholdImage(pred_img, 0.5)
            GT_label = sitk.GetImageFromArray(GT_array[:, :, :, 0])
            pred_img = getLargestConnectedComponents(pred_img)
            pred_img.SetSpacing([0.5, 0.5, 0.5])
            GT_label.SetSpacing([0.5, 0.5, 0.5])

            # get boundingboxes
            bb_P = getBoundingBox(pred_img)
            bb_GT = getBoundingBox(GT_label)
            startZ = max(0, min(bb_P[2], bb_GT[2]) - 1)

            sizeZ = max(bb_P[2] + bb_P[5], bb_GT[2] + bb_P[5]) - startZ + 1

            apex_P = sitk.RegionOfInterest(pred_img, [168, 168, math.floor(sizeZ / 3)],
                                           [0, 0, startZ])
            apex_GT = sitk.RegionOfInterest(GT_label, [168, 168, math.floor(sizeZ / 3)],
                                            [0, 0, startZ])
            mid_P = sitk.RegionOfInterest(pred_img, [168, 168, math.floor(sizeZ / 3)],
                                          [0, 0, startZ + math.floor(sizeZ / 3)])
            mid_GT = sitk.RegionOfInterest(GT_label, [168, 168, math.floor(sizeZ / 3)],
                                           [0, 0, startZ + math.floor(sizeZ / 3)])
            base_P = sitk.RegionOfInterest(pred_img, [168, 168, math.floor(sizeZ / 3)],
                                           [0, 0, startZ + 2 * math.floor(sizeZ / 3)])
            base_GT = sitk.RegionOfInterest(GT_label, [168, 168, math.floor(sizeZ / 3)],
                                            [0, 0, startZ + 2 * math.floor(sizeZ / 3)])

            sitk.WriteImage(apex_P, 'apex_P.nrrd')
            sitk.WriteImage(apex_GT, 'apex_GT.nrrd')
            sitk.WriteImage(mid_P, 'mid_P.nrrd')
            sitk.WriteImage(mid_GT, 'mid_GT.nrrd')
            sitk.WriteImage(base_P, 'base_P.nrrd')
            sitk.WriteImage(base_GT, 'base_GT.nrrd')

            dice_apex = getDice(apex_P, apex_GT)
            avd_apex = relativeAbsoluteVolumeDifference(apex_P, apex_GT)
            [hausdorff_apex, avgDist_apex] = getBoundaryDistances(apex_P, apex_GT)

            dice_mid = getDice(mid_P, mid_GT)
            avd_mid = relativeAbsoluteVolumeDifference(mid_P, mid_GT)
            [hausdorff_mid, avgDist_mid] = getBoundaryDistances(mid_P, mid_GT)

            dice_base = getDice(base_P, base_GT)
            avd_base = relativeAbsoluteVolumeDifference(base_P, base_GT)
            [hausdorff_base, avgDist_base] = getBoundaryDistances(base_P, base_GT)

            csvwriter.writerow(
                ['Case' + str(i), dice_apex, dice_mid, dice_base, avd_apex, avd_mid, avd_base, hausdorff_apex,
                 hausdorff_mid, hausdorff_base, avgDist_apex, avgDist_mid, avgDist_base])

            print(i)


def thresholdArray(array, threshold):
    # threshold image
    array[array < threshold] = 0
    array[array >= threshold] = 1
    array = np.asarray(array, np.int16)

    return array


def removeSegmentationsInImagePaddedRegion(array_test, array_pred):
    for i in range(0, array_test.shape[0]):
        if np.count_nonzero(array_test[i, :, :, 0]) == 0:
            array_pred[:, i, :, :] = 0


def getConnectedComponents(predictionImage):
    pred_img = castImage(predictionImage, sitk.sitkInt8)
    pred_img_cc = getLargestConnectedComponents(pred_img)
    pred_img_cc = castImage(pred_img_cc, sitk.sitkInt8)

    img_isl = sitk.Subtract(pred_img, pred_img_cc)

    return pred_img_cc, img_isl


def removeIslands(predictedArray):
    pred = predictedArray
    print(pred.shape)
    pred_pz = thresholdArray(pred[:, :, :, 0], 0.5)
    pred_cz = thresholdArray(pred[:, :, :, 1], 0.5)
    pred_us = thresholdArray(pred[:, :, :, 2], 0.5)
    pred_afs = thresholdArray(pred[:, :, :, 3], 0.5)
    pred_bg = thresholdArray(pred[:, :, :, 4], 0.5)

    pred_pz_img = sitk.GetImageFromArray(pred_pz)
    pred_cz_img = sitk.GetImageFromArray(pred_cz)
    pred_us_img = sitk.GetImageFromArray(pred_us)
    pred_afs_img = sitk.GetImageFromArray(pred_afs)
    pred_bg_img = sitk.GetImageFromArray(pred_bg)
    # pred_bg_img = utils.castImage(pred_bg, sitk.sitkInt8)

    pred_pz_img_cc, pz_otherCC = getConnectedComponents(pred_pz_img)
    pred_cz_img_cc, cz_otherCC = getConnectedComponents(pred_cz_img)
    pred_us_img_cc, us_otherCC = getConnectedComponents(pred_us_img)
    pred_afs_img_cc, afs_otherCC = getConnectedComponents(pred_afs_img)
    pred_bg_img_cc, bg_otherCC = getConnectedComponents(pred_bg_img)

    added_otherCC = sitk.Add(afs_otherCC, pz_otherCC)
    added_otherCC = sitk.Add(added_otherCC, cz_otherCC)
    added_otherCC = sitk.Add(added_otherCC, us_otherCC)
    added_otherCC = sitk.Add(added_otherCC, bg_otherCC)

    # sitk.WriteImage(added_otherCC, 'addedOtherCC.nrrd')
    # sitk.WriteImage(pred_cz_img, 'pred_cz.nrrd')

    pz_dis = sitk.SignedMaurerDistanceMap(pred_pz_img_cc, insideIsPositive=True, squaredDistance=False,
                                          useImageSpacing=False)
    cz_dis = sitk.SignedMaurerDistanceMap(pred_cz_img_cc, insideIsPositive=True, squaredDistance=False,
                                          useImageSpacing=False)
    us_dis = sitk.SignedMaurerDistanceMap(pred_us_img_cc, insideIsPositive=True, squaredDistance=False,
                                          useImageSpacing=False)
    afs_dis = sitk.SignedMaurerDistanceMap(pred_afs_img_cc, insideIsPositive=True, squaredDistance=False,
                                           useImageSpacing=False)
    bg_dis = sitk.SignedMaurerDistanceMap(pred_bg_img_cc, insideIsPositive=True, squaredDistance=False,
                                          useImageSpacing=False)

    # sitk.WriteImage(pred_cz_img_cc, 'pred_cz_cc.nrrd')
    # sitk.WriteImage(cz_dis, 'cz_dis.nrrd')

    array_pz = sitk.GetArrayFromImage(pred_pz_img_cc)
    array_cz = sitk.GetArrayFromImage(pred_cz_img_cc)
    array_us = sitk.GetArrayFromImage(pred_us_img_cc)
    array_afs = sitk.GetArrayFromImage(pred_afs_img_cc)
    array_bg = sitk.GetArrayFromImage(pred_bg_img_cc)

    finalPrediction = np.zeros([32, 168, 168, 5])
    finalPrediction[:, :, :, 0] = array_pz
    finalPrediction[:, :, :, 1] = array_cz
    finalPrediction[:, :, :, 2] = array_us
    finalPrediction[:, :, :, 3] = array_afs
    finalPrediction[:, :, :, 4] = array_bg

    for x in range(0, pred_cz_img.GetSize()[0]):
        for y in range(0, pred_cz_img.GetSize()[1]):
            for z in range(0, pred_cz_img.GetSize()[2]):

                pos = [x, y, z]
                if (added_otherCC[pos] > 0):
                    # print(pz_dis.GetPixel(x,y,z),cz_dis.GetPixel(x,y,z),us_dis.GetPixel(x,y,z), afs_dis.GetPixel(x,y,z))
                    array = [pz_dis.GetPixel(x, y, z), cz_dis.GetPixel(x, y, z), us_dis.GetPixel(x, y, z),
                             afs_dis.GetPixel(x, y, z), bg_dis.GetPixel(x, y, z)]
                    maxValue = max(array)
                    max_index = array.index(maxValue)
                    finalPrediction[z, y, x, max_index] = 1

    return finalPrediction


def postprocesAndEvaluateFiles(outDir, prediction, GT_array, csvName, eval=True, prediction_arr_exists=False,
                               model_name=None):
    if not prediction_arr_exists:
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        out_arr = np.zeros((prediction.shape[0], 32, 168, 168, 5))
        for i in range(0, prediction.shape[0]):
        #for i in range(0, 5):
            print(i)
            array = removeIslands(prediction[i, :, :, :, :])
            print(array.shape)
            out_arr[i] = array
        #npy_name = outDir + model_name
        #np.save(npy_name, out_arr.astype('int8'))

    if eval:
        evaluateFiles_zones(GT_array, prediction_arr=out_arr, csvName=csvName)
    return out_arr.astype('int8')


def predict_for_uats_mc(val_x_arr, val_y_arr, model, mc=False):
    val_supervised_flag = np.ones((val_x_arr.shape[0], 32, 168, 168), dtype='int8')

    x_val = [val_x_arr, val_y_arr, val_supervised_flag]
    print('load_weights')
    # train.load_weights()
    print('predict')
    out = model.predict(x_val, batch_size=1, verbose=1)
    output_arr = np.zeros((out[0].shape[0], 32, 168, 168, 5))

    output_arr[:, :, :, :, 0] = out[0]
    output_arr[:, :, :, :, 1] = out[1]
    output_arr[:, :, :, :, 2] = out[2]
    output_arr[:, :, :, :, 3] = out[3]
    output_arr[:, :, :, :, 4] = out[4]
    print(output_arr.shape)

    return output_arr


def generate_predictions(model_dir, model_name, ul_imgs):
    csvName = model_dir + model_name + '.csv'
    from dataset_specific.prostate.model.baseline import weighted_model
    wm = weighted_model()
    model = wm.build_model()
    model.load_weights(os.path.join(model_dir , model_name + '.h5'))
    out = model.predict(ul_imgs, batch_size=1, verbose=1)
    output_arr = np.zeros((out[0].shape[0], 32, 168, 168, 5))
    output_arr[:, :, :, :, 0] = out[0]
    output_arr[:, :, :, :, 1] = out[1]
    output_arr[:, :, :, :, 2] = out[2]
    output_arr[:, :, :, :, 3] = out[3]
    output_arr[:, :, :, :, 4] = out[4]
    print(output_arr.shape)

    return postprocesAndEvaluateFiles(model_dir, output_arr, None, eval=False, csvName=csvName, prediction_arr_exists=False,
                               model_name=model_name)


def evaluate_uats(model_dir, model_name, val_x, val_y, mc=False):
    csvName = model_dir + model_name + '.csv'

    if mc:
        from dataset_specific.prostate.model.uats_mc_dropout import weighted_model
        wm = weighted_model()
        model = wm.build_model(trained_model=model_dir + model_name)[0]
        model.load_weights(model_dir + model_name)
        prediction_arr = predict_for_uats_mc(val_x, val_y, model, mc=True)

    else:
        from dataset_specific.prostate.model.uats_scaled import weighted_model
        wm = weighted_model()
        model = wm.build_model(trained_model=model_dir + model_name, temp=1.0)
        model.load_weights(model_dir + model_name)
        prediction_arr = predict_for_uats_mc(val_x, val_y, model, mc=False)

    postprocesAndEvaluateFiles(model_dir, prediction_arr, val_y, eval=True, csvName=csvName,
                               prediction_arr_exists=False, model_name=model_name)


def evaluate_supervised(model_dir, model_name, val_x, val_y, eval=True):
    csvName = model_dir + model_name + '.csv'

    from dataset_specific.prostate.model.baseline import weighted_model
    wm = weighted_model()
    model = wm.build_model()
    model.load_weights(model_dir + model_name + '.h5')
    out = model.predict(val_x, batch_size=1, verbose=1)
    output_arr = np.zeros((out[0].shape[0], 32, 168, 168, 5))
    output_arr[:, :, :, :, 0] = out[0]
    output_arr[:, :, :, :, 1] = out[1]
    output_arr[:, :, :, :, 2] = out[2]
    output_arr[:, :, :, :, 3] = out[3]
    output_arr[:, :, :, :, 4] = out[4]
    print(output_arr.shape)
    return postprocesAndEvaluateFiles(model_dir, output_arr, val_y, eval=eval, csvName=csvName,
                               prediction_arr_exists=False, model_name=model_name)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    evaluate_uats(model_dir='/data/suhita/experiments/model/uats/prostate/',
                  # model_dir='/data/suhita/prostate/',
                  #model_name='uats_softmax_no_consistency_F4_Perct_Labelled_1.0.h5',
                 model_name='uats_mc_entropy_F1_Perct_Labelled_1.0.h5',

                  #model_name='uats_softmax_F2_Perct_Labelled_1.0.h5',
                  val_x=np.load('/cache/suhita/data/prostate/final_test_array_imgs.npy'),
                  val_y=np.load('/cache/suhita/data/prostate/final_test_array_GT.npy').astype('int8'),
                  mc=True,

                  )

    # arr = evaluate_supervised(model_dir='/data/suhita/experiments/model/prostate/',
    #                model_name='supervised_F2_P1.0',
    #                val_x=np.load('/cache/suhita/data/prostate/final_test_array_imgs.npy'),
    #                val_y=np.load('/cache/suhita/data/prostate/final_test_array_GT.npy').astype('int8')
    #                )
    # np.save('/data/suhita/experiments/model/prostate/supervised2.npy', arr)