import os
import cv2
import sys

def make_frames(input_file):
    vidcap = cv2.VideoCapture(input_file)
    success,image = vidcap.read()
    if(not success):
        print("Tried reading non-existant file!")
        return None
    count = 0
    os.mkdir(f'./frames')
    while success:
        image = cv2.resize(image, (1280,720))
        cv2.imwrite(f'frames/frames_{count:06}.jpg', image)   
        success,image = vidcap.read()
        count += 1
        if count % 10000 == 0:
            print("10000 frames compeleted")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        make_frames(sys.argv[1])
    else:
        print('specify input video')
