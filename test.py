import numpy as np

img = np.load('D:/Thesis/numpy/val/imgs/10.npy')
gt = np.load('D:/Thesis/numpy/val/gt/10.npy')

print(img.shape)
print(gt.shape)
slice = 8
import matplotlib.pyplot as plt

plt.imshow(img[slice, :, :, 0], cmap='coolwarm', alpha=1.)
plt.imshow(gt[slice, :, :, 0], alpha=0.4, cmap='copper')
plt.imshow(gt[slice, :, :, 1], alpha=0.4, cmap='Blues')
plt.imshow(gt[slice, :, :, 2], alpha=0.4, cmap='Oranges')
plt.imshow(gt[slice, :, :, 3], alpha=0.4, cmap='Greens')
plt.show()

plt.imshow(img[slice, :, :, 0], cmap='coolwarm', alpha=1.)
plt.show()
