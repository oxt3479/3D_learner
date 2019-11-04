import torch
import torch.nn as nn
import torch.nn.functional as F

class depthencoder(nn.Module):
    def __init__(self):
        super().__init__()
        encode = []
        DEVICE = torch.device("cuda:0")
        relu = nn.ReLU(True)
        pool = nn.MaxPool2d(2)
        
        encode.append(nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), relu, pool, nn.BatchNorm2d(32)))
        encode.append(nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), relu, pool, nn.BatchNorm2d(64)))
        encode.append(nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), relu, pool, nn.BatchNorm2d(128)))
        encode.append(nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), relu, pool, nn.BatchNorm2d(256)))
        encode.append(nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), relu, nn.BatchNorm2d(256)))
        
        self.encoders = []
        self.encoders.append(nn.Sequential(*encode).to(DEVICE)) # for n
        self.encoders.append(nn.Sequential(*encode).to(DEVICE)) # for n-1 & n+1
        #self.encoders.append(nn.Sequential(*encode).to(DEVICE)) # for D(n-1) & D(n+1)
        
        decode = []
        decode.append(nn.Sequential(nn.Conv2d(256*3, 256, 3, padding = 1), relu, nn.BatchNorm2d(256)))
        decode.append(nn.Sequential(nn.ConvTranspose2d(256*4, 128, 3, stride=2, \
                                                     padding=1, output_padding=1), relu, nn.BatchNorm2d(128)))
        decode.append(nn.Sequential(nn.ConvTranspose2d(128*4, 64, 3, stride=2, \
                                                     padding=1, output_padding=1), relu, nn.BatchNorm2d(64)))
        decode.append(nn.Sequential(nn.ConvTranspose2d(64*4, 32, 3, stride=2, \
                                                     padding=1, output_padding=1), relu, nn.BatchNorm2d(32)))
        decode.append(nn.Sequential(nn.ConvTranspose2d(32*4, 1, 3, stride=2, \
                                                     padding=1, output_padding=1)))
        self.decoder = nn.Sequential(*decode).to(DEVICE)
        todepth = []
        todepth.append(nn.ConvTranspose2d(128, 1, 3, padding = 1).to(DEVICE))
        todepth.append(nn.ConvTranspose2d(64, 1, 3, padding = 1).to(DEVICE))
        todepth.append(nn.ConvTranspose2d(32, 1, 3, padding = 1).to(DEVICE))
        self.todepth = todepth # Layers to convert intermediate results into maps
        '''
        transform = []
        transform.append(nn.Sequential(nn.Conv3d(1, 32, 3), relu))
        transform.append(nn.Sequential(nn.Conv2d(32, 1, 3), relu))
        
        self.transformer = nn.Sequential(*transform)
        '''    
    def forward(self, x1, x2, x3):
        xout = []
        inputs = [x1,x2,x3]
        skips = [[],[],[]]
        for i in range(3):
            x = inputs[i]
            for layer in self.encoders[i-1]: #Encoder 2 used for n-1 and n+1 (x1,x3)
                x = layer(x)
                skips[i].append(x)
        x = skips[-1][-1]
        for i, layer in enumerate(self.decoder):
            for j, encoder in enumerate(skips):
                if j == len(skips)-1 and i == 0: break
                x = torch.cat((x,encoder[-(i+1)]),1)
            x = layer(x)
            if(i > 0):
                if(i == 4):
                    xout.append(x)
                else:
                    #depth_get = nn.Sequential(self.todepth[:i-1])
                    #xout.append(depth_get(x))
                    xout.append(self.todepth[i-1](x))
        skips = []
        return xout