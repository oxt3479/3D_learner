#%%
import cv2
import numpy as np
import os

def build_data(directory):
    '''Creates memmap objects of all npy
    files in the given directory. For successful utilization 
    the directory should contain numpy files created with 
    mov2frames.py and frames2nparray.py,
    directory should ONLY contain these files.'''
    data = []
    for file in os.listdir(directory):
        numpy_file = np.load(directory+file, allow_pickle=True, mmap_mode = 'r+')
        data.append(numpy_file)
    return data


data = build_data("numpy_img/")

output = cv2.calcOpticalFlowFarneback(data[3][0,0,:,:,0], \
    data[3][0,1,:,:,0], None, 0.5, 3, 15, 3, 5, 1.2, 0) 

import matplotlib.pyplot as plt

plt.imshow(output[:,:,0])
plt.show()

# %%
