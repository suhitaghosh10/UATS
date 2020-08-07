import glob
import os
import shutil

import cv2
import numpy as np

IMG_WIDTH = 256
IMG_HEIGHT = 192


def generateFolds(directory, foldDir, nrSplits=5):

    from sklearn.model_selection import KFold
    if not os.path.isdir(foldDir):
        os.makedirs(foldDir)
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

def hair_removal(in_dir, out_dir):
    import cv2
    import numpy as np

    imgs = os.listdir(in_dir)

    # rgb = Image.fromarray(np.load("/cache/anneke/skin/preprocessed/labelled/train/imgs/ISIC_0010487.npy"), mode='RGB')

    for img in imgs:

        #src = cv2.imread(os.path.join(in_dir, img))
        src = np.load(os.path.join(in_dir, img))

        print(src.shape)

        # imshow(src)
        # show()
        # Convert the original_classification image to grayscale
        grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        # imshow(grayScale)
        # show()
        #cv2.imwrite('/data/suhita/skin/grayScale_' + name + '.jpg', grayScale, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # Kernel for the morphological filtering
        kernel = cv2.getStructuringElement(1, (17, 17))

        # Perform the blackHat filtering on the grayscale image to find the
        # hair countours
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        # imshow(blackhat)
        # show()
        # cv2.imwrite('/data/suhita/skin/blackhat_' + name + '.jpg', blackhat, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # intensify the hair countours in preparation for the inpainting
        # algorithm
        ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        print(thresh2.shape)
        # imshow(thresh2)
        # show()
        #cv2.imwrite('/data/suhita/skin/thresholded' + name + '.jpg', thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # inpaint the original_classification image depending on the mask
        dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)
        print('jj')
        # imshow(dst)
        # show()
        print('jj')
        np.save(os.path.join(out_dir, img), dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


if __name__ == '__main__':

   # create_and_move_test_images('/data/anneke/skin/labelled', out_dir='/data/anneke/skin/labelled/test')
   # create_and_move_test_images('/data/anneke/skin/labelled_GT', out_dir='/data/anneke/skin/labelled_GT/test', filetype='png')

   #preprocess_dir('/data/anneke/skin/unlabelled/','/data/anneke/skin/preprocessed/unlabelled', filetype = 'jpg')
   #preprocess_dir('/data/anneke/skin/labelled_GT/test', '/cache/anneke/skin/preprocessed/labelled/test/GT', filetype='png')

    # if not os.path.exists('Folds'):
    #     os.makedirs('Folds')

    #generateFolds('/home/anneke/data/skin/preprocessed/labelled/train/imgs', 'Folds_new', nrSplits=10)

    hair_removal('/home/anneke/data/skin/preprocessed/labelled/test/imgs', '/home/anneke/data/skin_less_hair/preprocessed/labelled/test/imgs')
    hair_removal('/home/anneke/data/skin/preprocessed/labelled/train/imgs', '/home/anneke/data/skin_less_hair/preprocessed/labelled/train/imgs')
    hair_removal('/home/anneke/data/skin/preprocessed/unlabelled', '/home/anneke/data/skin_less_hair/preprocessed/unlabelled')

   #
    #     #
    #     # cv2.imshow('image', img)
    #     cv2.imshow('processed', img_proc)
    #     #
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()