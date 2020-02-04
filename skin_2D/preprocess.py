import cv2
import glob
import os
import glob
import shutil
import numpy as np

IMG_WIDTH = 256
IMG_HEIGHT = 192


def generateFolds(directory, foldDir, nrSplits=5):

    from sklearn.model_selection import KFold

    X = os.listdir(directory)
    kf = KFold(n_splits=nrSplits, shuffle=True, random_state=5)
    i = 1
    for train, test in kf.split(X):
        train_data = np.array(X)[train]
        test_data = np.array(X)[test]
        print('train', train_data.shape, train_data)
        print('test', test_data.shape, test_data)
        np.save(foldDir + '/train_fold' + str(i), train_data)
        np.save(foldDir + '/val_fold' + str(i), test_data)
        i = i + 1

def preprocess_image(image, width, height, interpolator= cv2.INTER_LINEAR):

    # resize image
    image = cv2.resize(image,(width,height), interpolation=interpolator)
    # normalize
    #image = image/255

    return image

def create_and_move_test_images(in_dir, out_dir, filetype = 'jpg'):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = sorted(glob.glob(in_dir+'/*'+filetype))
    np.random.seed(1234)
    files_idx = np.arange(0,len(files))
    np.random.shuffle(files_idx)

    for file_ix in files_idx[0:500]:
        filename = files[file_ix].split('/')[-1]
        shutil.move(files[file_ix], os.path.join(out_dir, filename))


def preprocess_dir(in_dir, out_dir, filetype ='jpg'):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = sorted(glob.glob(in_dir + '/*.'+filetype))

    for file in files:

        filename = file.split('/')[-1]

        img = cv2.imread(file)
        print(img.shape[0]/img.shape[1])

        if filetype == 'png':  # we assume that this is GT data
            img_proc = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            np.save(os.path.join(out_dir, filename[:-3] + 'npy'), img_proc[:, :, 0:1])
        else:
            img_proc = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
            np.save(os.path.join(out_dir, filename[:-3] + 'npy'), img_proc)


if __name__ == '__main__':

   # create_and_move_test_images('/data/anneke/skin/labelled', out_dir='/data/anneke/skin/labelled/test')
   # create_and_move_test_images('/data/anneke/skin/labelled_GT', out_dir='/data/anneke/skin/labelled_GT/test', filetype='png')

   #preprocess_dir('/data/anneke/skin/unlabelled/','/data/anneke/skin/preprocessed/unlabelled', filetype = 'jpg')
   preprocess_dir('/data/anneke/skin/labelled_GT/test', '/cache/anneke/skin/preprocessed/labelled/test/GT', filetype='png')

    # if not os.path.exists('Folds'):
    #     os.makedirs('Folds')

    #generateFolds('/cache/anneke/skin/preprocessed/labelled/train/imgs', 'Folds', nrSplits=4)
   #
    #     #
    #     # cv2.imshow('image', img)
    #     cv2.imshow('processed', img_proc)
    #     #
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()