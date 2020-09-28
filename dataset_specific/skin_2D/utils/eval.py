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


def eval_for_uats_softmax(model_dir, model_name, batch_size=1, out_dir=None):
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
    model = wm.build_model(img_shape=(DIM[1], DIM[2], DIM[3]), learning_rate=1e-07, gpu_id=None,
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
    DIM = img_arr.shape
    from dataset_specific.skin_2D.model.baseline import weighted_model
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[1], DIM[2], DIM[3]), learning_rate=1e-07)
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
                          'uats_softmax_F3_Perct_Labelled_1.0', batch_size=1,
                          out_dir='/data/suhita/experiments/temp/')

    # eval_for_uats_mc(
    #     '/data/suhita/skin/models/',
    #    # '/data/suhita/temporal/skin/',
    #                        'sm_skin_sm_F3_Perct_Labelled_0.1',
    #                       batch_size=1, out_dir='/data/suhita/skin/eval/uats/', lesion=True, eval=True)

    # eval_for_supervised('/data/suhita/experiments/model/supervised/skin/', data_path,
    #                     'supervised_F1_P0.5', eval=True,
    #                     out_dir='/data/suhita/skin/ul/UL_' + str(perc), connected_component=True)
