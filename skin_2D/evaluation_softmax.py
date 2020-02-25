import csv

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from eval.preprocess import *

THRESHOLD = 0.5


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


def get_dice_from_array(arr1, arr2):
    y_true_f = arr1
    y_pred_f = arr2
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))


def evaluateFiles_arr(img_path, prediction, connected_component=False, eval=True, out_dir=None):
    nrImgs = prediction[0].shape[0]
    dices = np.zeros((nrImgs), dtype=np.float32)

    # print(dices.shape)

    for imgNumber in range(0, nrImgs):
        name = str(int(imgNumber))

        test_dir = sorted(os.listdir(os.path.join(img_path, 'imgs')))
        name = name + ' Case ' + test_dir[imgNumber]

        print(name)

        pred_bg = sitk.GetImageFromArray(thresholdArray(prediction[0][imgNumber], 0.5))
        pred_lesion = sitk.GetImageFromArray(thresholdArray(prediction[1][imgNumber], 0.5))
        pred_bg_img = castImage(pred_bg, sitk.sitkUInt8)
        pred_lesion_img = castImage(pred_lesion, sitk.sitkUInt8)

        if connected_component:
            pred_bg_img = getConnectedComponents(pred_bg_img)
            pred_lesion_img = getConnectedComponents(pred_lesion_img)

        # if not os.path.exists(out_dir + '/imgs'):
        #     os.makedirs(out_dir + '/imgs')
        # if not os.path.exists(out_dir + '/GT'):
        #     os.makedirs(out_dir + '/GT')

        pred_img_arr = np.stack((sitk.GetArrayFromImage(pred_bg_img), sitk.GetArrayFromImage(pred_lesion_img)), -1)
        # np.save(out_dir + '/imgs/' + str(imgNumber) + '.npy',
        #         np.load(os.path.join(img_path, 'imgs', test_dir[imgNumber])) / 255)
        # np.save(out_dir + '/GT/' + str(imgNumber) + '.npy', pred_img_arr)
        if eval:
            # lesion
            GT_label = np.load(
                os.path.join(img_path, 'GT', test_dir[imgNumber].replace('.npy', '_segmentation.npy'))) / 255
            dice = get_dice_from_array(pred_img_arr[:, :, 1], GT_label[:, :, 0])
            dices[imgNumber] = dice
            print(dice)


    print('Dices')
    print(np.average(dices))


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

    # img_isl = sitk.Subtract(pred_img, pred_img_cc)

    return pred_img_cc


def removeIslands(predictedArray):
    pred = predictedArray
    print(pred.shape)
    pred_pz = thresholdArray(pred[0, :, :, :], THRESHOLD)
    pred_cz = thresholdArray(pred[1, :, :, :], THRESHOLD)
    pred_us = thresholdArray(pred[2, :, :, :], THRESHOLD)
    pred_afs = thresholdArray(pred[3, :, :, :], THRESHOLD)
    pred_bg = thresholdArray(pred[4, :, :, :], THRESHOLD)

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

    finalPrediction = np.zeros([5, 32, 168, 168])
    finalPrediction[0] = array_pz
    finalPrediction[1] = array_cz
    finalPrediction[2] = array_us
    finalPrediction[3] = array_afs
    finalPrediction[4] = array_bg

    array = np.zeros([1, 1, 1, 1])

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
                    finalPrediction[max_index, z, y, x] = 1

    return finalPrediction


def create_test_arrays(test_dir, eval=True):
    cases = sorted(os.listdir(os.path.join(test_dir, 'imgs')))
    img = np.load(os.path.join(test_dir, 'imgs', cases[0]))
    img_arr = np.zeros((len(cases), img.shape[0], img.shape[1], 3), dtype=float)

    if eval:
        num = len(os.listdir(os.path.join(test_dir, 'GT')))
        GT_arr = np.zeros((num, img.shape[0], img.shape[1], 2), dtype=float)

    for i in range(len(cases)):
        if eval:
            GT_arr[i, :, :, 1] = np.load(os.path.join(test_dir, 'GT', cases[i]).replace('.npy', '_segmentation.npy'))[:,
                                 :, 0] / 255
            GT_arr[i, :, :, 0] = np.where(GT_arr[i, :, :, 1] == 0, np.ones_like(GT_arr[i, :, :, 1]),
                                          np.zeros_like(GT_arr[i, :, :, 1]))

        img_arr[i] = np.load(os.path.join(test_dir, 'imgs', cases[i])) / 255

    if eval:
        return img_arr, GT_arr
    else:
        return img_arr


def eval_for_uats_softmax(model_dir, model_name, batch_size=1, out_dir=None, connected_component=True):
    GT_dir = '/cache/suhita/skin/preprocessed/labelled/test/'
    print('create start')
    img_arr, GT_arr = create_test_arrays(GT_dir)
    print('create end')
    DIM = img_arr.shape
    from skin_2D.softmax.model_softmax_entropy import weighted_model
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[1], DIM[2], DIM[3]), learning_rate=learning_rate, gpu_id=None,
                           nb_gpus=None, trained_model=os.path.join(model_dir, model_name + '.h5'))
    # model.load_weights(os.path.join(model_dir, NAME,'.h5'))
    val_supervised_flag = np.ones((DIM[0], DIM[1], DIM[2], 1), dtype='int8')
    prediction = model[0].predict([img_arr, GT_arr, val_supervised_flag], batch_size=batch_size, verbose=1)
    csvName = os.path.join(model_dir, 'evaluation', model_name + '.csv')

    # weights epochs LR gpu_id dist orient prediction LRDecay earlyStop
    evaluateFiles_arr(img_path='/cache/suhita/skin/preprocessed/labelled/test/', prediction=prediction,
                      connected_component=True,
                      out_dir=out_dir, eval=True)


def eval_for_uats_mc(model_dir, model_name, batch_size=1, out_dir=None):
    GT_dir = '/cache/suhita/skin/preprocessed/labelled/test/'
    print('create start')
    img_arr, GT_arr = create_test_arrays(GT_dir)
    print('create end')
    DIM = img_arr.shape
    from skin_2D.softmax.model_softmax_entropy import weighted_model
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[1], DIM[2], DIM[3]), learning_rate=learning_rate, gpu_id=None,
                           nb_gpus=None, trained_model=os.path.join(model_dir, model_name + '.h5'))
    # model.load_weights(os.path.join(model_dir, NAME,'.h5'))
    val_supervised_flag = np.ones((DIM[0], DIM[1], DIM[2], 1), dtype='int8')
    prediction = model[0].predict([img_arr, GT_arr, val_supervised_flag], batch_size=batch_size, verbose=1)
    csvName = os.path.join(model_dir, 'evaluation', model_name + '.csv')

    # weights epochs LR gpu_id dist orient prediction LRDecay earlyStop
    evaluateFiles_arr(img_path='/cache/suhita/skin/preprocessed/labelled/test/',
                      prediction=prediction,
                      connected_component=True,
                      out_dir=out_dir, eval=True)


def eval_for_supervised(model_dir, img_path, model_name, eval=True, out_dir=None, connected_component=True):
    if eval:
        img_arr, GT_arr = create_test_arrays(img_path, eval=eval)
    else:
        img_arr = create_test_arrays(img_path, eval=eval)
        GT_arr = None
    DIM = img_arr.shape
    from skin_2D.softmax.model_softmax_baseline import weighted_model
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[1], DIM[2], DIM[3]), learning_rate=learning_rate)
    model.load_weights(os.path.join(model_dir, model_name + '.h5'))
    prediction = model.predict(img_arr, batch_size=1, verbose=1)

    # weights epochs LR gpu_id dist orient prediction LRDecay earlyStop
    evaluateFiles_arr(img_path=img_path, prediction=prediction, connected_component=connected_component,
                      out_dir=out_dir, eval=eval)


if __name__ == '__main__':
    # gpu = '/GPU:0'
    batch_size = 1


    # gpu = "GPU:0"  # gpu_id (default id is first of listed in parameters)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    learning_rate = 5e-5
    AUGMENTATION_NO = 5
    TRAIN_NUM = 50
    FOLD_NUM = 3
    augm = 'augm'
    batch_size = 2

    ### for baseline of 0.1 images,
    # NAME = 'supervised_F_centered_BB_' + str(FOLD_NUM) + '_' + str(TRAIN_NUM) + '_' + str(
    #     learning_rate) + '_Perc_' + str(PERC) + '_'+ augm
    perc = 0.5
    # model_dir = '/cache/suhita/skin/models/'
    # model_dir = '/data/suhita/temporal/skin/'
    # data_path = '/cache/suhita/skin/preprocessed/labelled/test/'
    # data_path = '/cache/suhita/skin/preprocessed/unlabelled/'
    # NAME = 'supervised_sfs32_F_1_1000_5e-05_Perc_' + str(perc) + '_augm'

    # eval_for_uats_softmax(model_dir, 'sm_skin_mc_F1_Perct_Labelled_' + str(perc), batch_size=1,
    #                    out_dir='/data/suhita/skin/ul_' + str(perc), connected_component=True)
    # eval_for_uats_mc('/data/suhita/temporal/skin/', 'sm_skin_mc_F1_Perct_Labelled_'+str(perc), batch_size=1, out_dir='/data/suhita/skin/eval')

    # eval_for_supervised('/data/suhita/skin/models/', data_path,
    #                     'softmax_supervised_sfs32_F_2_1000_5e-05_Perc_' + str(perc) + '_augm', eval=False,
    #                     out_dir='/data/suhita/skin/ul/UL_' + str(perc), connected_component=True)
# /data/suhita/temporal/skin/2_skin_softmax_F1_Perct_Labelled_1.0.h5

    data_path = '/home/anneke/data/skin_less_hair/preprocessed/labelled/test/'
    NAME = 'softmax_supervised_sfs32_F_'+str(FOLD_NUM)+'_1000_5e-05_Perc_' + str(perc) + '_augm'

    eval_for_supervised('output/models/', data_path,
                        NAME,  eval=True,
                        out_dir='/data/anneke', connected_component=True)