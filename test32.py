import numpy as np
import matplotlib.pyplot as plt

img = np.load("D:\\Thesis\\temp\\case_00231_old\\img_left.npy")
img_pp = np.load('D:\\Thesis\\temp\\case_00231\\img_left.npy')

gt = np.load("D:\\Thesis\\temp\\case_00231_old\\segm_left.npy")
gt_pp = np.load("D:\\Thesis\\temp\\case_00231\\segm_left.npy")
print(gt.shape)
print(img.shape)
slice_num = 14
plt.imshow(img[:, :, slice_num], alpha=1.0, cmap="Greys")
plt.show()
# plt.imshow(img[:,:,slice_num], alpha=1.0, cmap="Greys")
plt.imshow(gt[:, :, slice_num], alpha=0.5)
# plt.imshow(gt[8,:,:,1],alpha=0.5,cmap="Greens")
# plt.imshow(gt[8,:,:,2],alpha=0.5, cmap='Reds')
# plt.imshow(gt[8,:,:,3],alpha=0.5, cmap='Blues')
plt.show()

plt.imshow(img_pp[:, :, slice_num], alpha=1.0, cmap="Greys")
plt.show()
# plt.imshow(img_pp[:,:,slice_num], alpha=1.0, cmap="Greys")
plt.imshow(gt_pp[:, :, slice_num], alpha=0.5)
plt.show()
'''
a = gt[16,:,:,3]
b = np.where(a>0.5, a, np.zeros_like(a))
plt.imshow(img[16,:,:,0], alpha=1.0, cmap="Greys")
plt.imshow(b, cmap='coolwarm', alpha=0.7)
plt.show()

a = np.ravel(gt[16,:,:,3])
counter =0
sum=0
for i in np.arange(0,a.size):
    if a[i] > 0.5:
        sum = sum + a[i]
        counter = counter+1
print(sum/counter)
'''
