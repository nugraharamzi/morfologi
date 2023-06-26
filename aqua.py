import cv2   
import numpy as np
from skimage import data
from skimage.io import imread

import matplotlib.pyplot as plt

image = imread(fname="aqua2.jpg")

print(image.shape)
plt.imshow(image)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
      
# defining the range of masking 
blue1 = np.array([110, 50, 50]) 
blue2 = np.array([130, 255, 255]) 
      
# initializing the mask to be 
# convoluted over input image 
mask = cv2.inRange(hsv, blue1, blue2) 
  
# passing the bitwise_and over 
# each pixel convoluted 
res = cv2.bitwise_and(image, image, mask = mask) 
      
# defining the kernel i.e. Structuring element 
kernel = np.ones((5, 5), np.uint8) 
      
# defining the closing function  
# over the image and structuring element 
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 

fig, axes = plt.subplots(1, 2, figsize=(12, 12))
ax = axes.ravel()

ax[0].imshow(mask)
ax[0].set_title("Citra Input 1")

ax[1].imshow(closing, cmap='gray')
ax[1].set_title('Citra Input 2')

plt.show()