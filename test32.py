import numpy as np
import matplotlib.pyplot as plt

img_no = 'ISIC_0000311'
s_no = '57'
img = np.load("D:\\Thesis\\temp\\test\\imgs\\" + img_no + ".npy")
gt = np.load('D:\\Thesis\\temp\\test\\GT\\' + img_no + '_segmentation.npy')

plt.imshow(img, alpha=1.0)
plt.imshow(gt[:, :, 0], alpha=0.5)
plt.show()

plt.imshow(img, alpha=1.0)

plt.imshow(np.load('D:/Thesis/temp/test/pred/' + img_no + '.npy.npy')[:, :, 0], alpha=0.5)
plt.show()

img = np.load('D:\\Thesis\\temp\\test\\UL_0.05\\imgs\\' + s_no + '.npy')
gt = np.load('D:\\Thesis\\temp\\test\\UL_0.05\\GT\\' + s_no + '.npy'.replace('ISIC_', ''))

plt.imshow(img, alpha=1.0)
plt.imshow(gt[:, :, 0], alpha=0.5)
# plt.imshow(gt[8,:,:,1],alpha=0.5,cmap="Greens")
# plt.imshow(gt[8,:,:,2],alpha=0.5, cmap='Reds')
# plt.imshow(gt[8,:,:,3],alpha=0.5, cmap='Blues')
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
