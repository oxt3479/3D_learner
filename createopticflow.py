#%%
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

img = 53010
mov = 3

movie = os.listdir('./left')
path = f'./left/{movie[mov]}/'

image = os.listdir(path)[img]
previous = os.listdir(path)[img-1]
  
img = cv2.imread(path+image,  cv2.IMREAD_GRAYSCALE) 
color_img = cv2.imread(path+image) 
prev = cv2.imread(path+previous,  cv2.IMREAD_GRAYSCALE) 

output = cv2.calcOpticalFlowFarneback(img, prev, None, \
    0.5, 2, 15, 1, 5, 1.1, 0) 

plt.imshow(output[:,:,0])
plt.show()
plt.imshow(output[:,:,1])
plt.show()
plt.imshow(color_img, 'gray')
plt.show()

# %%
