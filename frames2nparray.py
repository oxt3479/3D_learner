# %%
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

REBUILD_DATA = True

class LeftToRight():
    IMG_SIZE = (1280,720) # Size of the movie frames
    LEFT = "left/"
    RIGHT = "right/"
    training_data = []
    size = 640, 256 # Size of NP arrays
    box = (0,100,1280,620) # Cropping of movie frames.
    def make_training_data(self, movie_details):
        LEFT = self.LEFT
        RIGHT = self.RIGHT
        size = self.size
        movie = movie_details[0]
        box = self.box
        # First build left 3 img dataset.
        k = 0
        j = 0
        for i in tqdm(range(movie_details[1], movie_details[2])):
            try:            
                #Reads in files from directories$ holding 720p frames of left and right stereo images
                path_L = os.path.join(LEFT+movie, os.listdir(LEFT+movie)[i])
                LN = Image.open(path_L).crop(box).resize(size, Image.ANTIALIAS)
                path_R = os.path.join(RIGHT+movie, os.listdir(RIGHT+movie)[i])
                RN = Image.open(path_R).crop(box).resize(size, Image.ANTIALIAS)
                self.training_data.append([np.array(LN),np.array(RN)])
            except Exception as e:
                print(str(e))
                pass
            j+=1
            if j == 5000 or i == movie_details[2]:
                k+=1
                np.save(f"{movie}_lowres-{k}.npy", self.training_data)
                self.training_data.clear()
                j = 0
        
if REBUILD_DATA:
    frame_ends = [['hobbit2p2', 0, 109000]]


    lefttoright = LeftToRight()
    for i in frame_ends:
        lefttoright.make_training_data(i)


    frame_ends = [['hobbit2', 1500, 108150],
        ['hobbit2p2', 0, 109000],
        ['hobbit3', 1500, 118350],
        ['hobbit3p2', 0, 70900],
        ['residentevilafterlife', 900, 129375],
        ['residentevilretribution', 1420, 123150],
        ['transformer_knight', 0, 210700],
        ['transformersmoon', 0, 210900],
        ['xmen_apocalypse', 1400, 193000]]

