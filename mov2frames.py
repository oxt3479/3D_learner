import os
import cv2
import sys

def make_frames(direc, movie):
    vidcap = cv2.VideoCapture(f'movies/{direc}/{movie} 3D-FS 720p.mkv')
    success,image = vidcap.read()
    count = 0
    if (os.path.isdir(f"/left/{movie}")):
        pass
    else: 
        os.mkdir(f"/left/{movie}")
        os.mkdir(f"/right/{movie}")
    while success:
        #sys.stdout.write("\r"+str(count))
        #sys.stdout.flush()
        image = cv2.resize(image, (1280,720))
        if count % 2 == 0:
            cv2.imwrite(f"left/{movie}/{movie}_%06d.jpg" % (count/2), image)   
        else:
            cv2.imwrite(f"right/{movie}/{movie}_%06d_R.jpg" % ((count-1)/2), image)
        success,image = vidcap.read()
        count += 1

if __name__ == "__main__":
    for(direc in os.listdir('movies/'))
        movie = os.listdir(f'movies/{direc}')[0]
        movie = movie.split(' ', 1)[0]
        make_frames(direc, movie)
        print(f"Completed movie: {movie}\n in directory: {direc}")

