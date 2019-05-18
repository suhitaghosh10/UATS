import math
import os
from shutil import copyfile

import SimpleITK as sitk
import numpy as np


############################ utils functions ##############################

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


# normlaize intensities according to the 99th and 1st percentile of the input image intensities
def normalizeIntensitiesPercentile(*imgs):
    i = 0
    for img in imgs:
        if i == 0:
            array = np.ndarray.flatten(sitk.GetArrayFromImage(img))
        else:
            array = np.append(array, np.ndarray.flatten(sitk.GetArrayFromImage(img)))
        i = i + 1

    upperPerc = np.percentile(array, 99)  # 98
    lowerPerc = np.percentile(array, 1)  # 2
    print(lowerPerc)

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    normalizationFilter = sitk.IntensityWindowingImageFilter()
    normalizationFilter.SetOutputMaximum(1.0)
    normalizationFilter.SetOutputMinimum(0.0)
    normalizationFilter.SetWindowMaximum(upperPerc)
    normalizationFilter.SetWindowMinimum(lowerPerc)

    out = []

    for img in imgs:
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


def resampleImage(inputImage, newSpacing, interpolator, defaultValue):
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
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


def resampleToReference(inputImg, referenceImg, interpolator, defaultValue):
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    inputImg = castImageFilter.Execute(inputImg)

    minFilter = sitk.StatisticsImageFilter()
    minFilter.Execute(inputImg)

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
            cz = 0
        else:
            correction = True
    else:
        cZ = 0

    # if correction is necessary, increase size of image with padding
    if correction:
        filter = sitk.ConstantPadImageFilter()
        print([math.ceil(cX / 2), math.ceil(cY), math.ceil(cZ / 2)])
        filter.SetPadLowerBound([int(math.floor(cX / 2)), int(math.floor(cY / 2)), int(math.floor(cZ / 2))])
        filter.SetPadUpperBound([int(math.ceil(cX / 2)), int(math.ceil(cY)), int(math.ceil(cZ / 2))])
        filter.SetConstant(-4)
        outPadding = filter.Execute(img)
        print('outPaddingSize', outPadding.GetSize())
        return outPadding

    else:
        return img


def getBoundingBox(img):
    masked = binaryThresholdImage(img, 0.1)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(masked)

    bb = statistics.GetBoundingBox(1)

    return bb


def getLargestConnectedComponents(img):
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedComponents = connectedFilter.Execute(img)

    labelStatistics = sitk.LabelShapeStatisticsImageFilter()
    labelStatistics.Execute(connectedComponents)
    nrLabels = labelStatistics.GetNumberOfLabels()

    biggestLabelSize = 0
    biggestLabelIndex = 1
    for i in range(1, nrLabels + 1):
        curr_size = labelStatistics.GetNumberOfPixels(i)
        if curr_size > biggestLabelSize:
            biggestLabelSize = curr_size
            biggestLabelIndex = i

    largestComponent = sitk.BinaryThreshold(connectedComponents, biggestLabelIndex, biggestLabelIndex)

    return largestComponent


def resample_array_segmentations_shapeBasedInterpolation(segm_array):
    upsampledArr = np.zeros([3, segm_array.shape[1], 168, 168, 168], dtype=np.uint8)
    segm_array = segm_array.astype(np.uint8)

    for i in range(0, segm_array.shape[1]):
        print(i)
        pz = sitk.GetImageFromArray(segm_array[0, i, :, :, :])

        cz = sitk.GetImageFromArray(segm_array[1, i, :, :, :])
        bg = sitk.GetImageFromArray(segm_array[2, i, :, :, :])
        spacing = [0.5, 0.5, 3.0]
        cz.SetSpacing(spacing)
        pz.SetSpacing(spacing)
        bg.SetSpacing(spacing)
        sitk.WriteImage(pz, 'pz.nrrd')

        pz_dis = sitk.SignedMaurerDistanceMap(pz, insideIsPositive=True, squaredDistance=False,
                                              useImageSpacing=True)
        pz_dis = resampleImage(pz_dis, [0.5, 0.5, 0.5], sitk.sitkLinear, -3000)
        pz_dis = sitk.DiscreteGaussian(pz_dis, variance=1.0)

        cz_dis = resampleToReference(sitk.SignedMaurerDistanceMap(cz, insideIsPositive=True, squaredDistance=False,
                                                                  useImageSpacing=True), pz_dis, sitk.sitkLinear, -3000)
        cz_dis = sitk.DiscreteGaussian(cz_dis, variance=1.0)
        sitk.WriteImage(cz_dis, 'cz_dist.nrrd')
        bg_dis = resampleToReference(sitk.SignedMaurerDistanceMap(bg, insideIsPositive=True, squaredDistance=False,
                                                                  useImageSpacing=True), pz_dis, sitk.sitkLinear, -3000)
        bg_dis = sitk.DiscreteGaussian(bg_dis, variance=1.0)
        sitk.WriteImage(bg_dis, 'bg_dist.nrrd')

        for x in range(0, 168):
            for y in range(0, 168):
                for z in range(0, 168):
                    if x == 83 and y == 96 and z == 96:
                        print('keks')
                    # print(pz_dis.GetPixel(x,y,z),cz_dis.GetPixel(x,y,z),us_dis.GetPixel(x,y,z), afs_dis.GetPixel(x,y,z))
                    array = [pz_dis.GetPixel(x, y, z), cz_dis.GetPixel(x, y, z), bg_dis.GetPixel(x, y, z)]
                    maxValue = max(array)
                    if maxValue == -3000:
                        upsampledArr[2, i, z, y, x] = 1
                    else:
                        max_index = array.index(maxValue)
                        # if max_index==0:
                        #     print('bla')
                        upsampledArr[max_index, i, z, y, x] = 1
                    # final_GT[z, y, x] = max_index+2
                    # print(x,y,z)
                    # print( maxValue)
    # final_GT_img = sitk.GetImageFromArray(final_GT)
    # final_GT_img = sitk.Threshold(final_GT_img, 2,5,0)
    # final_GT_img.CopyInformation(ref_image)

    return upsampledArr


# arbitrary main funtions
if __name__ == '__main__':

    outputDir = 'C:/Users/anneke/Documents/Boston/2017-11 ZonesData_onlyTra/'
    inputDir = 'D:/Daten/Prostate/Prostate Zones/zone segmentation slicer/'

    cases = os.listdir(inputDir)

    for case in cases:
        print(case)
        if not os.path.exists(outputDir + case):
            os.makedirs(outputDir + case)
        copyfile(inputDir + case + '/tra_upsampled.nrrd', outputDir + case + '/tra_upsampled.nrrd')
        copyfile(inputDir + case + '/Segmentation-label.nrrd', outputDir + case + '/Segmentation-label.nrrd')
