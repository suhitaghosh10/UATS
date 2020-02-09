import matplotlib.pyplot as plt
import numpy as np

img_no = 40
TRAIN_IMGS_PATH = np.load('D:\\Thesis\\temp\\img\\' + str(img_no) + '.npy')
TRAIN_GT_PATH = np.load('D:\\Thesis\\temp\\gt\\' + str(img_no) + '.npy')

slice_no = 16
zone = 0

plt.imshow(TRAIN_IMGS_PATH[slice_no, :, :, 0], cmap='gray')
plt.imshow(TRAIN_GT_PATH[slice_no, :, :, 3], alpha=0.4, cmap='viridis')
plt.imshow(TRAIN_GT_PATH[slice_no, :, :, 1], alpha=0.4, cmap='coolwarm')
# plt.imshow(TRAIN_GT_PATH[slice_no, :, :, 1], alpha=0.4, cmap='Reds')
# plt.imshow(TRAIN_GT_PATH[slice_no, :, :, 2], alpha=0.4, cmap='Greens')

plt.show()
