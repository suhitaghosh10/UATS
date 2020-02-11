# -----------------------------------------------------------------------------
# This file is created as part of the multi-planar prostate segmentation project
#
#  file:           utils.py
#  author:         Anneke Meyer, Otto-von-Guericke University Magdeburg
#  year:           2017
#
# -----------------------------------------------------------------------------


import os
import numpy as np
import math
import SimpleITK as sitk
import time
import json
import shutil


############################ utils functions ##############################

def log_time(start, name, d, n):
    end = time.time()
    # print(name, 'took ', str(end - start), 's.')
    d.append(end - start)
    n.append(name)
    return end


def save_experiment_json(misc, hyper_parameters, save_path):
    misc.update(hyper_parameters)
    with open(os.path.join(save_path, 'doc.json'), 'w') as fp:
        json.dump(misc, fp, indent=4)


def load_experiment_json(save_path):
    with open(os.path.join(save_path, 'doc.json'), 'r') as fp:
        data = json.load(fp)
    return data


def get_timestamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S")


def makeDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def getMeanAndStd(inputDir):
    patients = os.listdir(inputDir)
    list = []
    for patient in patients:
        data = os.listdir(inputDir + '/' + patient)
        for imgName in data:
            if 'tra' in imgName or 'cor' in imgName or 'sag' in imgName:
                img = sitk.ReadImage(inputDir + '/' + patient + '/' + imgName)
                arr = sitk.GetArrayFromImage(img)
                arr = np.ndarray.flatten(arr)

                list.append(np.ndarray.tolist(arr))

    array = np.concatenate(list).ravel()
    mean = np.mean(array)
    std = np.std(array)
    print(mean, std)
    return mean, std


def normalizeByMeanAndStd(img, mean, std):
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    img = castImageFilter.Execute(img)
    subFilter = sitk.SubtractImageFilter()
    image = subFilter.Execute(img, mean)

    divFilter = sitk.DivideImageFilter()
    image = divFilter.Execute(image, std)

    return image


def changeSizeWithPadding(inputImage, newSize):
    inSize = inputImage.GetSize()
    array = sitk.GetArrayFromImage(inputImage)
    minValue = array.min()

    print("newSize", newSize[0], newSize[1], newSize[2])

    nrVoxelsX = max(0, int((newSize[0] - inSize[0]) / 2))
    nrVoxelsY = max(0, int((newSize[1] - inSize[1]) / 2))
    nrVoxelsZ = max(0, int((newSize[2] - inSize[2]) / 2))

    if nrVoxelsX == 0 and nrVoxelsY == 0 and nrVoxelsZ == 0:
        print("sameSize")
        return inputImage

    print("Padding Constant Value", minValue)
    upperBound = [nrVoxelsX, nrVoxelsY, nrVoxelsZ]
    filter = sitk.ConstantPadImageFilter()
    filter.SetPadLowerBound([nrVoxelsX, nrVoxelsY, nrVoxelsZ])
    if inSize[0] % 2 == 1:
        upperBound[0] = nrVoxelsX + 1
    if inSize[1] % 2 == 1:
        upperBound[1] = nrVoxelsY + 1
    if inSize[2] % 2 == 1 and nrVoxelsZ != 0:
        upperBound[2] = nrVoxelsZ + 1

    filter.SetPadUpperBound(upperBound)
    filter.SetConstant(int(minValue))
    outPadding = filter.Execute(inputImage)
    print("outSize after Padding", outPadding.GetSize())

    return outPadding


# normlaize intensities according to the 99th and 1st percentile of the input image intensities
def normalizeIntensitiesPercentile(*imgs):
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


def getMaximumValue(img):
    maxFilter = sitk.StatisticsImageFilter()
    maxFilter.Execute(img)
    maxValue = maxFilter.GetMaximum()
    return maxValue


def thresholdImage(img, lowerValue, upperValue, outsideValue):
    thresholdFilter = sitk.ThresholdImageFilter()
    thresholdFilter.SetUpper(upperValue)
    thresholdFilter.SetLower(lowerValue)
    thresholdFilter.SetOutsideValue(outsideValue)

    out = thresholdFilter.Execute(img)
    return out


def binaryThresholdImage(img, lowerThreshold):
    maxFilter = sitk.StatisticsImageFilter()
    maxFilter.Execute(img)
    maxValue = maxFilter.GetMaximum()
    thresholded = sitk.BinaryThreshold(img, lowerThreshold, maxValue, 1, 0)

    return thresholded


def resampleImage(inputImage, newSpacing, interpolator, defaultValue, output_pixel_type = sitk.sitkFloat32):
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(output_pixel_type)
    inputImage = castImageFilter.Execute(inputImage)

    oldSize = inputImage.GetSize()
    oldSpacing = inputImage.GetSpacing()
    newWidth = oldSpacing[0] / newSpacing[0] * oldSize[0]
    newHeight = oldSpacing[1] / newSpacing[1] * oldSize[1]
    newDepth = oldSpacing[2] / newSpacing[2] * oldSize[2]
    newSize = [int(newWidth), int(newHeight), int(newDepth)]

    minFilter = sitk.StatisticsImageFilter()
    minFilter.Execute(inputImage)
    minValue = minFilter.GetMinimum()

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


def resampleToReference(inputImg, referenceImg, interpolator, defaultValue, out_dType= sitk.sitkFloat32):
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(out_dType)
    inputImg = castImageFilter.Execute(inputImg)

    # sitk.WriteImage(inputImg,'input.nrrd')

    filter = sitk.ResampleImageFilter()
    filter.SetReferenceImage(referenceImg)
    filter.SetDefaultPixelValue(float(defaultValue))  ## -1
    # float('nan')
    filter.SetInterpolator(interpolator)

    outImage = filter.Execute(inputImg)

    return outImage


def castImage(img, type):
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(type)  # sitk.sitkUInt8
    out = castFilter.Execute(img)

    return out


# corrects the size of an image to a multiple of the factor
def sizeCorrectionImage(img, factor, imgSize):
    # assumes that input image size is larger than minImgSize, except for z-dimension
    # factor is important in order to resample image by 1/factor (e.g. due to slice thickness) without any errors
    size = img.GetSize()
    correction = False
    # check if bounding box size is multiple of 'factor' and correct if necessary
    # x-direction
    if (size[0]) % factor != 0:
        cX = factor - (size[0] % factor)
        correction = True
    else:
        cX = 0
    # y-direction
    if (size[1]) % factor != 0:
        cY = factor - ((size[1]) % factor)
        correction = True
    else:
        cY = 0

    if (size[2]) != imgSize:
        cZ = (imgSize - size[2])
        # if z image size is larger than maxImgsSize, crop it (customized to the data at hand. Better if ROI extraction crops image)
        if cZ < 0:
            print('image gets filtered')
            cropFilter = sitk.CropImageFilter()
            cropFilter.SetUpperBoundaryCropSize([0, 0, int(math.floor(-cZ / 2))])
            cropFilter.SetLowerBoundaryCropSize([0, 0, int(math.ceil(-cZ / 2))])
            img = cropFilter.Execute(img)
            cZ = 0
        else:
            correction = True
    else:
        cZ = 0

    # if correction is necessary, increase size of image with padding
    if correction:
        filter = sitk.ConstantPadImageFilter()
        filter.SetPadLowerBound([int(math.floor(cX / 2)), int(math.floor(cY / 2)), int(math.floor(cZ / 2))])
        filter.SetPadUpperBound([math.ceil(cX / 2), math.ceil(cY), math.ceil(cZ / 2)])
        filter.SetConstant(0)
        outPadding = filter.Execute(img)
        return outPadding

    else:
        return img

def getMinimum(img):
    # get minimum value for padding
    filter = sitk.StatisticsImageFilter()
    filter.Execute(img)
    minValue = filter.GetMinimum()

    return minValue

def pad_volume(img, target_size_x=0, target_size_y=0, target_size_z=0, padValue = 0):

    act_size= img.GetSize()
    cX = 0
    cY = 0
    cZ = 0
    if target_size_x>0:
        cX = max(0,target_size_x - act_size[0])

    if target_size_y > 0:
        cY = max(0,target_size_y - act_size[1])

    if target_size_z > 0:
        cZ = max(0,target_size_z - act_size[2])


    filter = sitk.ConstantPadImageFilter()
    filter.SetPadLowerBound([int(math.floor(cX / 2)), int(math.floor(cY / 2)), int(math.floor(cZ / 2))])
    filter.SetPadUpperBound([math.ceil(cX / 2), math.ceil(cY / 2), math.ceil(cZ / 2)])
    filter.SetConstant(padValue)
    outPadding = filter.Execute(img)

    return outPadding


def resampleToReference_shapeBasedInterpolation(inputGT, referenceImage, backgroundValue=0):
    filter = sitk.SignedMaurerDistanceMapImageFilter()
    filter.SetUseImageSpacing(True)
    filter.UseImageSpacingOn()
    filter.SetInsideIsPositive(True)
    dist = filter.Execute(castImage(inputGT, sitk.sitkUInt8))
    dist = resampleToReference(dist, referenceImage, sitk.sitkLinear, -100)
    GT = binaryThresholdImage(dist, 0)

    return GT


def getBoundingBox(img):
    # masked = binaryThresholdImage(img, 0.1)
    ### ToDo: masked is needed for getting bounding box from intrsecting tra, sag, cor volumes. Is it disturing for other cases, e.g. evaluation?
    masked = binaryThresholdImage(img, 0.001)

    img = castImage(masked, sitk.sitkUInt8)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(img)

    bb = statistics.GetBoundingBox(1)

    return bb


def getLargestConnectedComponents(img):
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedComponents = connectedFilter.Execute(img)

    labelStatistics = sitk.LabelShapeStatisticsImageFilter()
    labelStatistics.Execute(connectedComponents)
    nrLabels = labelStatistics.GetNumberOfLabels()
    #print('Nr Of Labels:', nrLabels)

    if nrLabels == 1:
        return img

    biggestLabelSize = 0
    biggestLabelIndex = 1
    for i in range(1, nrLabels + 1):
        curr_size = labelStatistics.GetNumberOfPixels(i)
        if curr_size > biggestLabelSize:
            biggestLabelSize = curr_size
            biggestLabelIndex = i

    largestComponent = sitk.BinaryThreshold(connectedComponents, biggestLabelIndex, biggestLabelIndex)

    return largestComponent


def cropImage(img, lowerBound, upperBound):
    cropFilter = sitk.CropImageFilter()
    cropFilter.SetUpperBoundaryCropSize(upperBound)
    cropFilter.SetLowerBoundaryCropSize(lowerBound)
    img = cropFilter.Execute(img)

    return img


def makedir(dirpath):
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    try:
        os.makedirs(dirpath)
    except OSError:
        # [Errno 17] File exists
        pass


def generateFolds(directory, foldDir, nrSplits=5):
    from sklearn.model_selection import KFold

    X = os.listdir(directory)
    X = [x for x in X if '_deformed' not in x]
    kf = KFold(n_splits=nrSplits, shuffle=True, random_state=5)
    i = 1
    for train, test in kf.split(X):
        train_data = np.array(X)[train]
        test_data = np.array(X)[test]
        print('train', train_data.shape, train_data)
        print('test', test_data.shape, test_data)
        np.save(foldDir + '/train_fold' + str(i), train_data)
        np.save(foldDir + '/val_fold' + str(i), test_data)
        i = i + 1


if __name__ == '__main__':
    files_dir = '/data/anneke/kits_challenge/kits19/data/labelled/train'
    preprocessed_dir = '/data/anneke/kits_challenge/kits19/data/preprocessed/train'
    generateFolds(files_dir, 'Folds', nrSplits=4)
