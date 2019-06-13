import numpy as np

gt = np.load('D:/Thesis/weight_model/sl2/100g.npy')
img = np.load('D:/Thesis/weight_model/sl2/100.npy')

print(img.shape)
print(gt.shape)
slice = 16
zone = 0
import matplotlib.pyplot as plt

plt.imshow(img[slice, :, :, 0], cmap='Greys')
plt.imshow(gt[slice, :, :, zone], cmap='coolwarm', alpha=0.6)
plt.show()

plt.imshow(img[slice, :, :, 0], cmap='Greys', alpha=1.)
plt.show()
