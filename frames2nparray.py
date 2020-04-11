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


    def paths_in_npy_out(self, path_LM, path_LN, path_LO, path_R, box, size):
        try:
            LN = Image.open(path_LN).crop(box).resize(size, Image.ANTIALIAS)
            RN = Image.open(path_R).crop(box).resize(size, Image.ANTIALIAS)

            first = cv2.imread(path_LM,  cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(path_LN,  cv2.IMREAD_GRAYSCALE)
            last = cv2.imread(path_LO,  cv2.IMREAD_GRAYSCALE)
            
            LM_flow = cv2.calcOpticalFlowFarneback(img, first, None, 0.5, 2, 15, 1, 5, 1.1, 0)[100:620, :, :]
            LO_flow = cv2.calcOpticalFlowFarneback(img, last, None, 0.5, 2, 15, 1, 5, 1.1, 0)[100:620, :, :]
            LM_flow = cv2.resize(LM_flow, dsize=(640, 256))
            LO_flow = cv2.resize(LO_flow, dsize=(640, 256))

            LN_flow = np.concatenate((LM_flow, np.array(LN), LO_flow, np.array(RN)), axis=2)
            return LN_flow

        except Exception as e:
            print(str(e))
            return None



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
            #Reads in files from directories$ holding 720p frames of left and right stereo images
            path_LM = os.path.join(LEFT+movie, os.listdir(LEFT+movie)[i-1])
            path_LN = os.path.join(LEFT+movie, os.listdir(LEFT+movie)[i])
            path_LO = os.path.join(LEFT+movie, os.listdir(LEFT+movie)[i+1])
            path_R = os.path.join(RIGHT+movie, os.listdir(RIGHT+movie)[i])
            try:
                self.training_data.append(self.paths_in_npy_out(path_LM, path_LN, path_LO, path_R, box, size))
            except Exception as e:
                print(str(e))
                pass

            j+=1
            if j == 700 or i+1 == movie_details[2]:
                k+=1
                np.save(f"numpy_flow/{movie}_wflow-{k}.npy", self.training_data)
                self.training_data.clear()
                j = 0
        
if REBUILD_DATA:
    frame_ends = [['hobbit2', 1500, 108150]]


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



# %%
