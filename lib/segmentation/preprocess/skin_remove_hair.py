"""
Following are the DHR tasks followed in this example code:

    -- Applying Morphological Black-Hat transformation
    -- Creating the mask for InPainting task
    -- Applying inpainting algorithm on the image
"""

import cv2
import scipy.misc
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow, show

name = 'ISIC_0000102.jpg'

# rgb = Image.fromarray(np.load("/cache/anneke/skin/preprocessed/labelled/train/imgs/ISIC_0010487.npy"), mode='RGB')
src = cv2.imread('/data/suhita/skin/labelled/train/' + name)

print(src.shape)
# cv2.imshow("original Image", src)
imshow(src)
show()
# Convert the original image to grayscale
grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
imshow(grayScale)
show()
cv2.imwrite('/data/suhita/skin/grayScale_' + name + '.jpg', grayScale, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

# Kernel for the morphological filtering
kernel = cv2.getStructuringElement(1, (20, 20))

# Perform the blackHat filtering on the grayscale image to find the
# hair countours
blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
imshow(blackhat)
show()
cv2.imwrite('/data/suhita/skin/blackhat_' + name + '.jpg', blackhat, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

# intensify the hair countours in preparation for the inpainting
# algorithm
ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
print(thresh2.shape)
imshow(thresh2)
show()
cv2.imwrite('/data/suhita/skin/thresholded_' + name + '.jpg', thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

# inpaint the original image depending on the mask
dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)
print('jj')
imshow(dst)
show()
print('jj')
cv2.imwrite('/data/suhita/skin/InPainted' + name + '.jpg', dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
