import matplotlib.pyplot as plt
import numpy as np

img_no = 5
TRAIN_IMGS_PATH = np.load('D:/dataset/f3/val/imgs/' + str(img_no) + '.npy')
TRAIN_GT_PATH = np.load('D:/dataset/f3/val/gt/' + str(img_no) + '.npy')

slice_no = 16
zone = 0

plt.imshow(TRAIN_IMGS_PATH[slice_no, :, :, 0], cmap='gray')
plt.imshow(TRAIN_GT_PATH[slice_no, :, :, 3], alpha=0.4, cmap='viridis')
plt.imshow(TRAIN_GT_PATH[slice_no, :, :, zone], alpha=0.4, cmap='coolwarm')
plt.show()
