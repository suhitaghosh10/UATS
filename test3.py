import matplotlib.pyplot as plt
import numpy as np

TRAIN_IMGS_PATH = np.load('D:/dataset/f2/val/imgs/15.npy')
TRAIN_GT_PATH = np.load('D:/dataset/f2/val/gt/15.npy')

img_no = 1
slice_no = 16
zone = 0

plt.imshow(TRAIN_IMGS_PATH[slice_no, :, :, 0], cmap='gray')
plt.imshow(TRAIN_GT_PATH[slice_no, :, :, 3], alpha=0.4, cmap='viridis')
plt.imshow(TRAIN_GT_PATH[slice_no, :, :, zone], alpha=0.4, cmap='coolwarm')
plt.show()