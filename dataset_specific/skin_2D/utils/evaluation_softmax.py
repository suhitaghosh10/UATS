import csv

from dataset_specific.prostate.utils.preprocess import *

THRESHOLD = 0.5


def jaccard(prediction, groundTruth):
    filter = sitk.LabelOverlapMeasuresImageFilter()
    filter.Execute(prediction, groundTruth)
    return filter.GetJaccardCoefficient()


def getDice(prediction, groundTruth):
    filter = sitk.LabelOverlapMeasuresImageFilter()
    filter.Execute(prediction, groundTruth)
    return filter.GetDiceCoefficient()


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


def get_thresholded_jaccard(y_true, y_pred, smooth=1, axis=None):
    if axis is None:
        intersection = np.sum(np.abs(y_true * y_pred))
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        jaccard = np.mean((intersection + smooth) / (union + smooth))
    else:
        intersection = np.sum(np.abs(y_true * y_pred), axis=axis)
        union = np.sum(y_true, axis) + np.sum(y_pred, axis) - intersection
        jaccard = np.mean((intersection + smooth) / (union + smooth), axis=0)

    # return jaccard if jaccard>0.64 else 0
    return jaccard


def evaluateFiles_arr(img_path, prediction, connected_component=False, eval=True, out_dir=None, lesion=True):
    nrImgs = prediction[0].shape[0]
    dices = np.zeros((nrImgs), dtype=np.float32)
    jacs = np.zeros_like(dices)

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

        if not os.path.exists(out_dir + '/imgs'):
            os.makedirs(out_dir + '/imgs')
        if not os.path.exists(out_dir + '/GT'):
            os.makedirs(out_dir + '/GT')

        pred_img_arr = np.stack((sitk.GetArrayFromImage(pred_bg_img), sitk.GetArrayFromImage(pred_lesion_img)), -1)
        # np.save(out_dir + '/imgs/' + str(imgNumber) + '.npy',
        #         np.load(os.path.join(img_path, 'imgs', test_dir[imgNumber])) / 255)
        # np.save(out_dir + '/GT/' + str(imgNumber) + '.npy', pred_img_arr)
        if eval:
            # lesion
            GT_label = np.load(
                os.path.join(img_path, 'GT', test_dir[imgNumber].replace('.npy', '_segmentation.npy'))) / 255
            if lesion:
                dice = get_dice_from_array(pred_img_arr[:, :, 1], GT_label[:, :, 0])
                jac = get_thresholded_jaccard(pred_img_arr[:, :, 1], GT_label[:, :, 0])
                # jac = jac if jac > 0.64 else jac
            else:
                dice = get_dice_from_array(pred_img_arr[:, :, 0], 1 - GT_label[:, :, 0])
                jac = get_thresholded_jaccard(pred_img_arr[:, :, 0], 1 - GT_label[:, :, 0])
                # jac = jac if jac > 0.64 else jac
            dices[imgNumber] = dice
            jacs[imgNumber] = jac
            print(dice, jac)
            np.save(out_dir + '/imgs/' + test_dir[imgNumber],
                    np.load(os.path.join(img_path, 'imgs', test_dir[imgNumber])))
            np.save(out_dir + '/GT/' + test_dir[imgNumber], pred_img_arr)

        else:
            np.save(out_dir + '/imgs/' + test_dir[imgNumber],
                    np.load(os.path.join(img_path, 'imgs', test_dir[imgNumber])))
            np.save(out_dir + '/GT/' + test_dir[imgNumber], pred_img_arr)

    print('Dices')
    print(np.average(dices))
    print('Jaccard')
    print(np.average(jacs))


def evaluateFiles(GT_directory, pred_directory, csvName, lesion=False):
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


def create_test_arrays(test_dir, eval=True, save=False):
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
    from dataset_specific.skin_2D.model.uats_softmax import weighted_model
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[1], DIM[2], DIM[3]), learning_rate=5e-04, gpu_id=None,
                           nb_gpus=None, trained_model=os.path.join(model_dir, model_name + '.h5'))
    model.load_weights(os.path.join(model_dir, model_name + '.h5'))
    val_supervised_flag = np.ones((DIM[0], DIM[1], DIM[2]), dtype='int8')
    prediction = model.predict([img_arr, GT_arr, val_supervised_flag], batch_size=batch_size, verbose=1)

    # weights epochs LR gpu_id dist orient prediction LRDecay earlyStop
    evaluateFiles_arr(img_path='/cache/suhita/skin/preprocessed/labelled/test/', prediction=prediction,
                      connected_component=True,
                      out_dir=out_dir, eval=True)


def eval_for_uats_mc(model_dir, model_name, batch_size=1, out_dir=None, lesion=False, eval=True):
    GT_dir = '/cache/suhita/skin/preprocessed/labelled/test/'
    print('create start')
    img_arr, GT_arr = create_test_arrays(GT_dir)

    print('create end')
    DIM = img_arr.shape
    from dataset_specific.skin_2D.model.uats_entropy import weighted_model
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[1], DIM[2], DIM[3]), learning_rate=learning_rate, gpu_id=None,
                           nb_gpus=None, trained_model=os.path.join(model_dir, model_name + '.h5'))
    model[0].load_weights(os.path.join(model_dir, model_name + '.h5'))
    val_supervised_flag = np.ones((DIM[0], DIM[1], DIM[2], 1), dtype='int8')
    prediction = model[0].predict([img_arr, GT_arr, val_supervised_flag], batch_size=batch_size, verbose=1)

    # weights epochs LR gpu_id dist orient prediction LRDecay earlyStop
    evaluateFiles_arr(img_path='/cache/suhita/skin/preprocessed/labelled/test/',
                      prediction=prediction,
                      connected_component=True,
                      out_dir=out_dir, eval=eval, lesion=lesion)


def eval_for_supervised(model_dir, img_path, model_name, eval=True, out_dir=None, connected_component=True):
    if eval:
        img_arr, GT_arr = create_test_arrays(img_path, eval=eval)
    else:
        img_arr = create_test_arrays(img_path, eval=eval)
        GT_arr = None
    DIM = img_arr.shape
    from dataset_specific.skin_2D.model.baseline import weighted_model
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[1], DIM[2], DIM[3]), learning_rate=learning_rate)
    model.load_weights(os.path.join(model_dir, model_name + '.h5'))
    prediction = model.predict(img_arr, batch_size=1, verbose=1)

    # weights epochs LR gpu_id dist orient prediction LRDecay earlyStop
    evaluateFiles_arr(img_path=img_path, prediction=prediction, connected_component=connected_component,
                      out_dir=out_dir, eval=eval)




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    batch_size = 8
    data_path = '/cache/suhita/skin/preprocessed/labelled/test/'
    perc = 1.0
    FOLD_NUM = 1
    eval_for_uats_softmax('/data/suhita/experiments/model/uats/skin/',
                          'uats_softmax_F2_Perct_Labelled_1.0', batch_size=1,
                          out_dir='/data/suhita/experiments/temp/', connected_component=True)

    # eval_for_uats_mc(
    #     '/data/suhita/skin/models/',
    #    # '/data/suhita/temporal/skin/',
    #                        'sm_skin_sm_F3_Perct_Labelled_0.1',
    #                       batch_size=1, out_dir='/data/suhita/skin/eval/uats/', lesion=True, eval=True)

    # eval_for_supervised('/data/suhita/skin/models/', data_path,
    #                     'softmax_supervised_sfs32_F_2_1000_5e-05_Perc_' + str(perc) + '_augm', eval=True,
    #                     out_dir='/data/suhita/skin/ul/UL_' + str(perc), connected_component=True)
# /data/suhita/temporal/skin/2_skin_softmax_F1_Perct_Labelled_1.0.h5

# data_path = '/home/anneke/data/skin_less_hair/preprocessed/labelled/test/'
# NAME = 'softmax_supervised_sfs32_F_'+str(FOLD_NUM)+'_1000_5e-05_Perc_' + str(perc) + '_augm'

# from kits.utils import makedir
# perc = [0.1]
# for p in perc:
#     makedir('/data/suhita/skin/ul/UL_' + str(p))
#     # eval_for_supervised('/data/suhita/skin/models/', '/cache/suhita/skin/preprocessed/labelled/test/',
#     #              'softmax_supervised_sfs32_F_'+str(FOLD_NUM)+'_1000_5e-05_Perc_' + str(p) + '_augm', eval=True,
#     #              out_dir='/data/suhita/skin/ul/UL_' + str(p), connected_component=True)
#     eval_for_supervised('/data/suhita/skin/models/', '/cache/suhita/skin/preprocessed/unlabelled/',
#                          'softmax_supervised_sfs32_F_'+str(FOLD_NUM)+'_1000_5e-05_Perc_' + str(p) + '_augm', eval=False,
#                          out_dir='/data/suhita/skin/ul/UL_' + str(p), connected_component=True)
