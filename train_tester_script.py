# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import os

from monodepthloss import MonodepthLoss
from depthnet import *

DEVICE = torch.device("cuda:0")


# %%
encoderdecoder = ResnetModel(3).to(DEVICE)

optimizer = optim.Adam(encoderdecoder.parameters(),lr=0.0001)
loss_function = MonodepthLoss(n=4, SSIM_w=0.85,                disp_gradient_w=0.1, lr_w=1).to(DEVICE)

#encoderdecoder.load_state_dict(torch.load('state_dicts/encoderdecoder-1578671458'))
data = []
directory = "numpy_img/"
for file in os.listdir(directory):
    numpy_file = np.load(directory+file, allow_pickle=True, mmap_mode = 'r+')
    data.append(numpy_file)


# %%
testing = False
j = 0
mean = []
n = 12
epochs = 10


def test_model(testing_indeces):
    with torch.no_grad():
        mean = []
        for training_index in testing_indeces:
            random_num = random.randint(1,101)
            if random_num == 100:
                imageLEFT = torch.from_numpy(data[training_index[0]][training_index[0],0]).type(torch.cuda.FloatTensor)

                imageRIGHT = torch.from_numpy(data[training_index[0]][training_index[0],1]).type(torch.cuda.FloatTensor)

                inputLEFT = torch.div(imageLEFT, 255)

                inputRIGHT = torch.div(imageRIGHT, 255)

                output = encoderdecoder(inputLEFT.view(-1,3,256,640))
                loss = loss_function(output,[inputLEFT.view(-1,3,256,640), inputRIGHT.view(-1,3,256,640)])
                mean.append(loss.item())
    return round(sum(mean)/len(mean),5)

f= open(f"logs/results-{int(time.time())}.txt","w+")
for epoch in range(epochs):
    print("Epoch: "+ str(epoch))
    training_indeces = []
    testing_indeces = []
    for number, array in enumerate(data):
        frame_numbers = list(range(len(array)))
        random.shuffle(frame_numbers)
        for i in range(0, len(frame_numbers), n):
            frame_set = frame_numbers[i:i+n]
            random.shuffle(frame_set)
            if i >= len(frame_numbers)*.9:
                testing_indeces.append([number, frame_set])
            else:
                training_indeces.append([number, frame_set])
                # Make last 10% testing to ensure novelty
    random.shuffle(training_indeces)
     
    for training_index in tqdm(training_indeces):
        imageLEFT = torch.from_numpy(data[training_index[0]][training_index[0],0]).type(torch.cuda.FloatTensor)

        imageRIGHT = torch.from_numpy(data[training_index[0]][training_index[0],1]).type(torch.cuda.FloatTensor)

        inputLEFT = torch.div(imageLEFT, 255)

        inputRIGHT = torch.div(imageRIGHT, 255)

        encoderdecoder.zero_grad()
        output = encoderdecoder(inputLEFT.view(-1,3,256,640))
        loss = loss_function(output,[inputLEFT.view(-1,3,256,640), inputRIGHT.view(-1,3,256,640)])
        loss.backward()
        mean.append(loss.item())
        j += 1
        if j % 1000 == 0:
            trueloss = test_model(testing_indeces)
            f.write(f"{round(sum(mean)/len(mean),5)}, {trueloss}\n")
            f.flush()
            mean = []
            # Record the average training loss over time
        if j % 10000 == 0:
            thetime = int(time.time())
            torch.save(encoderdecoder.state_dict(), f"state_dicts/encoderdecoder-{thetime}")
        optimizer.step()
f.close()


# %%
plt.imshow(output[0][0,0,:,:].view(256, 640).cpu().detach().numpy(), 'gray')
plt.show()
plt.imshow(inputRIGHT[0,:,:,:].view(3,256,640).permute(1,2,0).cpu())
plt.show()
# See final output for training
