import numpy as np

arr = np.random.rand(60, 168, 168, 32, 1)
arr_ravel = np.ravel(arr)
print('unravel')
del arr
ind = np.argpartition(arr_ravel, -451584)[-451584:]
print('sort')
mask = np.ones(arr_ravel.shape, dtype=bool)
mask[ind] = False
print('num')
arr_ravel[mask] = 0
print('for')
arr = np.reshape(arr_ravel, (60, 168, 168, 32, 1))
print('end')
