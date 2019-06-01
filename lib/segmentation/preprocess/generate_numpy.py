import numpy as np

timgs = np.load('/home/suhita/zonals/data/training/trainArray_imgs_fold1.npy')
print(timgs.shape)
tgt = np.load('/home/suhita/zonals/data/training/trainArray_GT_fold1.npy')
tgt = tgt.astype('bool').astype('int8')

imgs = np.load('/home/suhita/zonals/data/training/trainArray_unlabeled_imgs_fold1.npy')
gt = np.load('/home/suhita/zonals/data/training/trainArray_unlabeled_GT_fold1.npy')
gt = gt.astype('bool').astype('int8')

stop = imgs.shape[0] + 58
print(stop)

for i in np.arange(timgs.shape[0]):
    np.save('/home/suhita/zonals/data/training/imgs/' + str(i), timgs[i])
    np.save('/home/suhita/zonals/data/training/gt/' + str(i), tgt[i])
    print(i)

for i in np.arange(start=58, stop=stop):
    np.save('/home/suhita/zonals/data/training/imgs/' + str(i), imgs[i - 58])
    np.save('/home/suhita/zonals/data/training/gt/' + str(i), gt[i - 58])
    print(i)
