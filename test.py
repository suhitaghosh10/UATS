import SimpleITK as sitk
import numpy as np


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


for i in np.arange(25):
    if i < 10:
        a, b, c = load_itk('D:/Thesis/Promise12/TrainingData_Part1/Case' + str(0) + str(i) + '.mhd')
    else:
        a, b, c = load_itk('D:/Thesis/Promise12/TrainingData_Part1/Case' + str(i) + '.mhd')
    print(c)
