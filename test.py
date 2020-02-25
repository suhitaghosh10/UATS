from matplotlib.pyplot import plot, imshow, show
import numpy as np
import cv2
from PIL import Image

img_arr = np.load('D:/Thesis/img.npy')
lesion_arr = np.load('D:/Thesis/lesion.npy')
print(img_arr.shape)
print(lesion_arr.shape)

img_no = 274
imshow(img_arr[img_no], alpha=1.0)
kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])
rgb = Image.fromarray(img_arr[img_no], mode='RGB')
grayScale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
sharpened = cv2.filter2D(grayScale, -1, kernel)
imshow(sharpened, alpha=1.0)
show()

imshow(img_arr[img_no], alpha=1.0)
imshow(lesion_arr[1, img_no], alpha=0.3)
show()
