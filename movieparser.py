import cv2
import torch
DEVICE = torch.device("cuda:0")

def getframeset(frames, images):
    imageLNm = []
    imageLN = []
    imageLNp = []
    imageRNm = []
    imageRN = []
    imageRNp = []
    for frame in frames:
        LNm = cv2.imread('left/skyscraper/'+images[frame-2], cv2.IMREAD_GRAYSCALE)
        LN = cv2.imread('left/skyscraper/'+images[frame-1], cv2.IMREAD_GRAYSCALE)
        LNp = cv2.imread('left/skyscraper/'+images[frame], cv2.IMREAD_GRAYSCALE)
        RNm = cv2.imread('right/skyscraper/'+images[frame-2][0:-4]+'_R.jpg', cv2.IMREAD_GRAYSCALE)
        RN = cv2.imread('right/skyscraper/'+images[frame-1][0:-4]+'_R.jpg', cv2.IMREAD_GRAYSCALE)
        RNp = cv2.imread('right/skyscraper/'+images[frame][0:-4]+'_R.jpg', cv2.IMREAD_GRAYSCALE)
        imageLNm.append(LNm)
        imageLN.append(LN)
        imageLNp.append(LNp)
        imageRNm.append(RNm)
        imageRN.append(RN)
        imageRNp.append(RNp)
        
        
    
    LinNm = torch.as_tensor(imageLNm,dtype=float).view(-1,1,1280,720).to(DEVICE)
    LinN = torch.as_tensor(imageLN,dtype=float).view(-1,1,1280,720).to(DEVICE)
    LinNp = torch.as_tensor(imageLNp,dtype=float).view(-1,1,1280,720).to(DEVICE)

    RoutNm = torch.as_tensor(imageRNm,dtype=float).view(-1,1,1280,720).to(DEVICE)
    RoutN = torch.as_tensor(imageRN,dtype=float).view(-1,1,1280,720).to(DEVICE)
    RoutNp = torch.as_tensor(imageRNp,dtype=float).view(-1,1,1280,720).to(DEVICE)

    return LinNm, LinN, LinNp, RoutNm, RoutN, RoutNp