import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

REBUILD_DATA = True

class inference_input():
    IMG_SIZE = (1280,720) # Size of the movie frames
    DIR = "frames/"
    training_data = []
    size = 640, 256 # Size of NP arrays
    box = (0,100,1280,620) # Cropping of movie frames.


    def paths_in_npy_out(self, path_LM, path_LN, path_LO, box, size):
        try:
            LN = Image.open(path_LN).crop(box).resize(size, Image.ANTIALIAS)

            first = cv2.imread(path_LM,  cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(path_LN,  cv2.IMREAD_GRAYSCALE)
            last = cv2.imread(path_LO,  cv2.IMREAD_GRAYSCALE)
            
            LM_flow = cv2.calcOpticalFlowFarneback(img, first, None, 0.5, 2, 15, 1, 5, 1.1, 0)[100:620, :, :]
            LO_flow = cv2.calcOpticalFlowFarneback(img, last, None, 0.5, 2, 15, 1, 5, 1.1, 0)[100:620, :, :]
            LM_flow = cv2.resize(LM_flow, dsize=(640, 256))
            LO_flow = cv2.resize(LO_flow, dsize=(640, 256))
            LN_flow = np.concatenate((LM_flow, np.array(LN), LO_flow, np.zeros((256,640,3))), axis=2)
            return LN_flow

        except Exception as e:
            print(str(e))
            return None



    def make_data(self):
        DIR = self.DIR
        size = self.size
        box = self.box
        k = 0
        j = 0
        MAX = len(os.listdir(DIR))
        for i, frame in tqdm(enumerate(os.listdir(DIR))):
            #Reads in files from directories$ holding 720p frames of left and right stereo images
            path_LM = os.path.join(DIR, os.listdir(DIR)[i-1])
            path_LN = os.path.join(DIR, os.listdir(DIR)[i])
            path_LO = os.path.join(DIR, os.listdir(DIR)[i+1])
            try:
                self.training_data.append(self.paths_in_npy_out(path_LM, path_LN, path_LO, box, size))
            except Exception as e:
                print(str(e))
                pass
            j+=1
            if j == 700 or i+1 == MAX:
                k+=1
                np.save(f"PREDICT_flow/pred_wflow-{k:04}.npy", self.training_data)
                self.training_data.clear()
                j = 0
LOTR = inference_input()
LOTR.make_data()

