import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

img_no = 40
trn = 'ROTATE'
img = sitk.GetArrayFromImage(sitk.ReadImage('D:/thesis/temp/test/changed_imagehippocampus_001_' + trn + '.nrrd'))
gt0 = sitk.GetArrayFromImage(sitk.ReadImage('D:/thesis/temp/test/ch_gthippocampus_001_0_' + trn + '.nrrd'))
gt1 = sitk.GetArrayFromImage(sitk.ReadImage('D:/thesis/temp/test/ch_gthippocampus_001_1_' + trn + '.nrrd'))
gt2 = sitk.GetArrayFromImage(sitk.ReadImage('D:/thesis/temp/test/ch_gthippocampus_001_2_' + trn + '.nrrd'))

slice_no = 24
zone = 0
plt.imshow(img[:, :, slice_no], alpha=1, cmap='Greys')
plt.show()
plt.imshow(img[:, :, slice_no], alpha=1, cmap='Greys')
plt.imshow(gt0[:, :, slice_no], alpha=0.4, cmap='Greens')
plt.imshow(gt1[:, :, slice_no], alpha=0.4, cmap='Reds')
plt.imshow(gt2[:, :, slice_no], alpha=0.4, cmap='Blues')


plt.show()
