import matplotlib.pyplot as plt
import numpy as np

imgs = np.load('D:/Thesis/numpy/test/final_test_array_imgs.npy')
gt = np.load('D:/Thesis/numpy/test/final_test_array_GT.npy')
pgt = np.empty((20, 32, 168, 168, 5))
notsgt = np.empty((20, 32, 168, 168, 5))
temp = np.load('D:/Thesis/numpy/test/predicted_sl2.npy')

temp = np.transpose(temp, (1, 2, 3, 4, 0))
pgt[:, :, :, :, 0] = temp[:, :, :, :, 0]
pgt[:, :, :, :, 1] = temp[:, :, :, :, 1]
pgt[:, :, :, :, 2] = temp[:, :, :, :, 2]
pgt[:, :, :, :, 3] = temp[:, :, :, :, 3]
pgt[:, :, :, :, 4] = temp[:, :, :, :, 4]

notsgt = np.load('D:/Thesis/numpy/test/predicted_final_5p.npy')
print(notsgt.shape)

print(gt.shape)
img = 15
slice = 16
z = 3
plt.imshow(imgs[img, slice, :, :, 0], cmap='Greys')
plt.imshow(gt[img, 16, :, :, z], alpha=0.6, cmap='OrRd_r')
plt.imshow(pgt[img, slice, :, :, z], alpha=0.6, cmap='coolwarm')
plt.show()
