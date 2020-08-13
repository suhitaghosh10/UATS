import numpy as np

npz = np.load('data/cifar10.npz')
images = npz.f.train_x
labels = npz.f.train_y

print(np.unique(labels))
print('images -', images.shape)
print('label -', labels.shape)

import matplotlib.pyplot as plt

plt.imshow(images[20, :, :, :])
plt.show()

print(images[20, :, :, :].shape)
trs = np.pad(images[20, :, :, :], ((2, 2), (2, 2), (0, 0)), 'reflect')
plt.imshow(trs)
plt.show()
print(trs.shape)
