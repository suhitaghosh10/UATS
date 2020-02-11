import csv

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from hippocampus.baseline import get_multi_class_arr
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


def evaluateFiles_arr(prediction, GT_array, csvName, connected_component= False):
    with open(csvName, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';', lineterminator='\n',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(
            ['Case', 'Class0 Dice', 'Class0 MeanDis', 'Class0 95-HD',
             'Class1 Dice', 'Class1 MeanDis', 'Class1 95-HD',
             'Class2 Dice', 'Class2 MeanDis', 'Class2 95-HD'])

        nrImgs = GT_array.shape[0]
        dices = np.zeros((nrImgs, 3), dtype=np.float32)
        print(dices.shape)
        mad = np.zeros((nrImgs, 3), dtype=np.float32)
        hdf = np.zeros((nrImgs, 3), dtype=np.float32)
        # auc = np.zeros((nrImgs, 4), dtype=np.float32)

        for imgNumber in range(0, nrImgs):

            if connected_component:
                prediction_temp = removeIslands(np.asarray(prediction)[:,imgNumber, :, :, :])
            else:
                prediction_temp = np.asarray(prediction)[:,imgNumber, :,:,:]
            values = ['Case' + str(imgNumber)]
            print('Case' + str(int(imgNumber)))
            for class_idx in range (0, 3):

                pred_arr = prediction_temp[class_idx, :,:,:]
                pred_arr = thresholdArray(pred_arr, 0.5)
                pred_img = sitk.GetImageFromArray(pred_arr)

                GT_label = sitk.GetImageFromArray(GT_arr[imgNumber,:,:,:,class_idx])
                GT_label.SetSpacing([1.0,1.0,1.0])
                pred_img = castImage(pred_img, sitk.sitkUInt8)
                pred_img.SetSpacing([1.0, 1.0, 1.0])
                GT_label = castImage(GT_label, sitk.sitkUInt8)


                dice = getDice(pred_img, GT_label)

                # avd = relativeAbsoluteVolumeDifference(pred_img, GT_label)
                [hausdorff, avgDist] = getBoundaryDistances(pred_img, GT_label)



                dices[imgNumber, class_idx] = dice
                mad[imgNumber, class_idx] = avgDist
                hdf[imgNumber, class_idx] = hausdorff
                #auc[imgNumber, zoneIndex] = roc_auc

                values.append(dice)
                values.append(avgDist)
                values.append(hausdorff)

            csvwriter.writerow(values)

        csvwriter.writerow('')
        average = ['Average', np.average(dices[:,0]),
                   np.average(mad[:,0]), np.average(hdf[:,0]), np.average(dices[:,1]),
                   np.average(mad[:,1]), np.average(hdf[:,1]),np.average( dices[:,2]),
                   np.average(mad[:,2]), np.average(hdf[:,2])]
        median = ['Median', np.median(dices[:,0]), np.median(mad[:,0]), np.median(hdf[:,0]), np.median(dices[:,1]),
                   np.median(mad[:,1]), np.median(hdf[:,1]),np.median( dices[:,2]),
                   np.median(mad[:,2]), np.median(hdf[:,2])]
        std = ['STD', np.std(dices[:,0]), np.std(mad[:,0]), np.std(hdf[:,0]), np.std(dices[:,1]),
                   np.std(mad[:,1]), np.std(hdf[:,1]),np.std( dices[:,2]),
                   np.std(mad[:,2]), np.std(hdf[:,2])]

        csvwriter.writerow(average)
        csvwriter.writerow(median)
        csvwriter.writerow(std)

        print('Dices')
        print(np.average(dices))

        print('Mean Dist')
        print(np.average(mad))


        print('Hausdorff 95%')
        print(np.average(hdf))



def thresholdArray(array, threshold):
    # threshold image
    array[array < threshold] = 0
    array[array >= threshold] = 1
    array = np.asarray(array, np.int16)

    return array



def getConnectedComponents(predictionImage):
    pred_img = castImage(predictionImage, sitk.sitkInt8)
    pred_img_cc = getLargestConnectedComponents(pred_img)
    pred_img_cc = castImage(pred_img_cc, sitk.sitkInt8)

    img_isl = sitk.Subtract(pred_img, pred_img_cc)

    return pred_img_cc, img_isl


def removeIslands(predictedArray):
    pred = predictedArray
    print(pred.shape)
    pred_bg = thresholdArray(pred[0, :, :, :], THRESHOLD)
    pred_c1 = thresholdArray(pred[1, :, :, :], THRESHOLD)
    pred_c2 = thresholdArray(pred[2, :, :, :], THRESHOLD)

    pred_bg_img = sitk.GetImageFromArray(pred_bg)
    pred_c1_img = sitk.GetImageFromArray(pred_c1)
    pred_c2_img = sitk.GetImageFromArray(pred_c2)
    # pred_bg_img = utils.castImage(pred_bg, sitk.sitkInt8)

    pred_bg_img_cc, bg_otherCC = getConnectedComponents(pred_bg_img)
    pred_c1_img_cc, c1_otherCC = getConnectedComponents(pred_c1_img)
    pred_c2_img_cc, c2_otherCC = getConnectedComponents(pred_c2_img)

    added_otherCC = sitk.Add(bg_otherCC, c1_otherCC)
    added_otherCC = sitk.Add(added_otherCC, c2_otherCC)

    sitk.WriteImage(added_otherCC, 'addedOtherCC.nrrd')
    sitk.WriteImage(pred_c1_img_cc, 'pred_c1_img_cc.nrrd')
    sitk.WriteImage(pred_c2_img_cc, 'pred_c2_img_cc.nrrd')
    sitk.WriteImage(pred_bg_img_cc, 'pred_bg_img_cc.nrrd')

    bg_dis = sitk.SignedMaurerDistanceMap(pred_bg_img_cc, insideIsPositive=True, squaredDistance=False,
                                          useImageSpacing=False)
    c1_dis = sitk.SignedMaurerDistanceMap(pred_c1_img_cc, insideIsPositive=True, squaredDistance=False,
                                          useImageSpacing=False)
    c2_dis = sitk.SignedMaurerDistanceMap(pred_c2_img_cc, insideIsPositive=True, squaredDistance=False,
                                          useImageSpacing=False)



    # sitk.WriteImage(pred_cz_img_cc, 'pred_cz_cc.nrrd')
    # sitk.WriteImage(cz_dis, 'cz_dis.nrrd')

    array_c1 = sitk.GetArrayFromImage(pred_c1_img_cc)
    array_c2 = sitk.GetArrayFromImage(pred_c2_img_cc)
    array_bg = sitk.GetArrayFromImage(pred_bg_img_cc)

    finalPrediction = np.zeros([3,48,64,48])
    finalPrediction[0] = array_bg
    finalPrediction[1] = array_c1
    finalPrediction[2] = array_c2

    array = np.zeros([1, 1, 1, 1])

    for x in range(0, pred_bg_img.GetSize()[0]):
        for y in range(0, pred_bg_img.GetSize()[1]):
            for z in range(0, pred_bg_img.GetSize()[2]):

                pos = [x, y, z]
                if (added_otherCC[pos] > 0):
                    # print(pz_dis.GetPixel(x,y,z),cz_dis.GetPixel(x,y,z),us_dis.GetPixel(x,y,z), afs_dis.GetPixel(x,y,z))
                    array = [c1_dis.GetPixel(x, y, z), c2_dis.GetPixel(x, y, z), bg_dis.GetPixel(x, y, z)]
                    maxValue = max(array)
                    max_index = array.index(maxValue)
                    finalPrediction[max_index, z, y, x] = 1

    return finalPrediction



def create_test_arrays(test_dir_img, test_dir_labels, n_classes=1):

    cases = sorted(os.listdir(test_dir_img))
    cases = [x for x in cases if '.npy' in x]

    # open first case to obtain dimensions
    segm = np.load(os.path.join(test_dir_img, cases[0]))
    DIM = segm.shape
    img_arr = np.zeros((len(cases) , DIM[0], DIM[1], DIM[2], 1), dtype=float)
    GT_arr = np.zeros((len(cases), DIM[0], DIM[1], DIM[2], n_classes), dtype=float)


    for i in range(len(cases)):
        img_arr[i, :, :, :, 0] = np.load(os.path.join(test_dir_img, cases[i]))
        GT_arr[i] = get_multi_class_arr(np.load(os.path.join(test_dir_labels, cases[i])),
                                                            n_classes=3)

    return img_arr, GT_arr

if __name__ == '__main__':

    from hippocampus.model import weighted_model

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    learning_rate = 4e-5
    AUGMENTATION_NO = 5
    TRAIN_NUM = 150
    PERCENTAGE = [0.05, 0.1, 0.25, 0.5, 1.0]
    FOLD_NUM = 2
    augm = 'augm'
    batch_size = 2

    for PERC in PERCENTAGE:

        NAME = 'supervised_F_' + str(FOLD_NUM) + '_' + str(TRAIN_NUM) + '_' + str(
            learning_rate) + '_Perc_' + str(PERC) + '_'+ augm


        out_dir = '/home/anneke/projects/uats/code/hippocampus/output'
        model_dir = out_dir +'/models'

        GT_dir_imgs = '/cache/anneke/hippocampus/preprocessed/labelled/test'
        GT_dir_labels = '/cache/anneke/hippocampus/preprocessed/labelled-GT/test'

        img_arr, GT_arr = create_test_arrays(GT_dir_imgs, GT_dir_labels, n_classes=3)
        DIM = img_arr.shape
        wm = weighted_model()

        model = wm.build_model(img_shape=(DIM[1], DIM[2], DIM[3]), learning_rate=5e-5)
        model.load_weights(os.path.join(model_dir, NAME+'.h5'))
        prediction = model.predict(img_arr, batch_size=batch_size)



        csvName =os.path.join(out_dir, 'evaluation_CC', NAME+'.csv')


        # weights epochs LR gpu_id dist orient prediction LRDecay earlyStop
        evaluateFiles_arr(prediction=prediction, GT_array=GT_arr, csvName=csvName, connected_component= True)
