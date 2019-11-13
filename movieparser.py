import torch
DEVICE = torch.device("cuda:0")
from PIL import Image
from numpy import *
from torchvision import transforms

def getframeset(frames, images):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        ])#normalize])
    imageLNm = []
    imageLN = []
    imageLNp = []
    imageRNm = []
    imageRN = []
    imageRNp = []
    size = 640, 256
    for frame in frames:
        LNmb = Image.open('left/skyscraper/'+images[frame-2])
        LNb = Image.open('left/skyscraper/'+images[frame-1])
        LNpb = Image.open('left/skyscraper/'+images[frame])
        RNmb = Image.open('right/skyscraper/'+images[frame-2][0:-4]+'_R.jpg')
        RNb = Image.open('right/skyscraper/'+images[frame-1][0:-4]+'_R.jpg')
        RNpb = Image.open('right/skyscraper/'+images[frame][0:-4]+'_R.jpg')
        
        LNm = LNmb.resize(size, Image.ANTIALIAS)
        LN = LNb.resize(size, Image.ANTIALIAS)
        LNp = LNpb.resize(size, Image.ANTIALIAS)
        RNm = RNmb.resize(size, Image.ANTIALIAS)
        RN = RNb.resize(size, Image.ANTIALIAS)
        RNp = RNmb.resize(size, Image.ANTIALIAS)
        LinNm = preprocess(LNm).view(-1,3,640,256).to(DEVICE)
        LinN = preprocess(LN).view(-1,3,640,256).to(DEVICE)
        LinNp = preprocess(LNp).view(-1,3,640,256).to(DEVICE)
        RoutNm = preprocess(RNm).view(-1,3,640,256).to(DEVICE)
        RoutN = preprocess(RN).view(-1,3,640,256).to(DEVICE)
        RoutNp = preprocess(RNp).view(-1,3,640,256).to(DEVICE)
        imageLNm.append(LinNm)
        imageLN.append(LinN)
        imageLNp.append(LinNp)
        imageRNm.append(RoutNm)
        imageRN.append(RoutN)
        imageRNp.append(RoutNp)
    out1 = torch.cat(imageLNm,0)
    out2 = torch.cat(imageLN,0)
    out3 = torch.cat(imageLNp,0)
    out4 = torch.cat(imageRNm,0)
    out5 = torch.cat(imageRN,0)
    out6 = torch.cat(imageRNp,0)
    return out1, out2, out3, out4, out5, out6