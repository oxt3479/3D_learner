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
import sys
from monodepthloss import MonodepthLoss
from depthnet import *
from PIL import Image


class trainer_tester:
    def __init__(self, dict_name=None):
        super().__init__()
        self.DEVICE = torch.device("cuda:0")
        self.encoderdecoder = ResnetModel(7).to(self.DEVICE)
        self.optimizer = optim.Adam(self.encoderdecoder.parameters(),lr=0.001)
        self.loss_function = MonodepthLoss(n=4, SSIM_w=0.85, disp_gradient_w=0.1, lr_w=1).to(self.DEVICE)
        if dict_name != None:
            self.encoderdecoder.load_state_dict(torch.load(f'state_dicts/{dict_name}'))
            print(f'Network initialized with {dict_name}')
        else:
            print('No weights input')
        self.data = self.build_data("numpy_flow/")
        self.n = 12
        # N is batch number, i.e. number of frames per itteration
        self.epochs = 10


    def build_data(self, directory):
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


    def test_model(self, testing_indeces):
        '''Function takes testing_indeces that
        reference data and determines the loss without
        calculating gradients (i.e. doesn't train on test data)
        testing_indeces: [(int) npy file index in data directory, [([int])frame indeces]]
        returns: calculated loss
        '''
        val_mean = []
        with torch.no_grad(): 
            for testing_index in testing_indeces:
                random_num = random.randint(1,160) # Sample ~100 samples (so variance matches between testing and training means)
                if random_num == 1: # (Only 1/160th of the training data is tested, randomly)
                    # Take the images out of the data variable and apply simple transformation
                    inputLEFT, inputRIGHT = self.get_input_arrays(testing_index)
                    # Use the left image to generate a loss
                    output = self.encoderdecoder(inputLEFT.view(-1,7,256,640))
                    val_loss = self.loss_function(output,[inputLEFT.view(-1,7,256,640), inputRIGHT.view(-1,3,256,640)])
                    val_mean.append(val_loss.item())
        return round(sum(val_mean)/(.01+len(val_mean)),5)


    def get_data_indeces(self):
        '''Creates pair of list of .npy memmap 
        indeces paired with sets of n (batch size) 
        frame numbers, randomly assorted.'''
        data = self.data
        training_indeces = []
        testing_indeces = []
        for number, array in enumerate(data):
            frame_numbers = list(range(len(array)))
            random.shuffle(frame_numbers)
            for i in range(0, len(frame_numbers), self.n):
                frame_set = frame_numbers[i:i+self.n]
                random.shuffle(frame_set)
                if i >= len(frame_numbers)*.9:
                    testing_indeces.append([number, frame_set])
                else:
                    training_indeces.append([number, frame_set])
                    # Make last 10% testing to ensure novelty
        random.shuffle(training_indeces)
        return training_indeces, testing_indeces


    def get_input_arrays(self, training_index):
        '''With an index (from a list of indeces generated by get_data_indeces)
        return the two npy arrays they index and transform them for training input.
        imageLEFT contains a 3 channel image sandwiched between two optic flow layers.'''
        data = self.data
        imageLEFT = torch.from_numpy(data[training_index[0]][training_index[1],:,:,0:7]).type(torch.cuda.FloatTensor)
        imageRIGHT = torch.from_numpy(data[training_index[0]][training_index[1],:,:,7:10]).type(torch.cuda.FloatTensor)
        imageLEFT[:,:,:,0:2] = torch.div(imageLEFT[:,:,:,0:2], torch.mean(abs(imageLEFT[:,:,:,0:2]))*2)
        imageLEFT[:,:,:,2:5] = torch.div(imageLEFT[:,:,:,2:5], 255)
        imageLEFT[:,:,:,5:7] = torch.div(imageLEFT[:,:,:,5:7], torch.mean(abs(imageLEFT[:,:,:,5:7]))*2)
        # Scale the optic flow channels so that the average optic flow becomes ~ .5
        inputLEFT = imageLEFT.permute(0,3,1,2)
        inputRIGHT = torch.div(imageRIGHT, 255).permute(0,3,1,2)
        # Convert uint8 inputs into floats between 0 and 1, permute channels so color/optic-flow is second
        return inputLEFT, inputRIGHT


    def display_results(self, movie, frame, name=None):
        '''Graphs two arrays, of network input and output 
        specified by a npy index number movie, and frame integer.'''
        data = self.data
        with torch.no_grad():
            imageLEFT = torch.from_numpy(data[movie][frame,0]).type(torch.cuda.FloatTensor)
            inputLEFT = torch.div(imageLEFT, 255).permute(2,0,1)
            output = self.encoderdecoder(inputLEFT.view(-1,3,256,640))
        result = output[0][0,0,:,:].view(256, 640).cpu().detach().numpy()
        
        fig = plt.figure(figsize=(16,4))
        fig.patch.set_visible(False)

        ax0 = fig.add_subplot(121)
        ax0.imshow(imageLEFT[:,:,:].view(256,640,3).cpu()/255)

        ax1 = fig.add_subplot(122)
        ax1.imshow(np.clip(result, -.02, .02))

        plt.tight_layout()
        ax0.axis('off')
        ax1.axis('off')
        if name != None:
            plt.savefig(f'examples/{name}_test.png')
        plt.show()
        return None
    

    def render_framerange(self, movie, frame0, frame1, name):
        '''
        Outputs pngs over a provided framerange for the source movie and its depth result.
        '''
        data = self.data
        with torch.no_grad():
            for frame in range(frame0, frame1):
                imageLEFT = torch.from_numpy(data[movie][frame,:,:,0:7]).type(torch.cuda.FloatTensor)
                imageLEFT[:,:,0:2] = torch.div(imageLEFT[:,:,0:2], torch.mean(abs(imageLEFT[:,:,0:2]))*2)
                imageLEFT[:,:,2:5] = torch.div(imageLEFT[:,:,2:5], 255)
                imageLEFT[:,:,5:7] = torch.div(imageLEFT[:,:,5:7], torch.mean(abs(imageLEFT[:,:,5:7]))*2)
                inputLEFT = imageLEFT.permute(2,0,1)
                output = self.encoderdecoder(inputLEFT.view(-1,7,256,640))
                result = output[0][0,0,:,:].view(256, 640).cpu().detach().numpy()
                im_result = (np.clip(result, -.02, .02)+.02)*255*25
                im_result = im_result.astype(np.uint8)
                im = Image.fromarray(im_result)
                im.convert('L')
                left = Image.fromarray(data[movie][frame,:,:,2:5].astype(np.uint8))
                im.save(f'depth_mov/d_{name}_{frame:05}.png') 
                left.save(f'input_mov/{name}_{frame:05}.png')
                if name == 'optic':
                    flow = data[movie][frame,:,:,5] + 20
                    Image.fromarray(flow.astype(np.uint8)).save(f'input_mov/0_{name}_{frame:05}.png')


    def train(self):
        data = self.data
        n = self.n
        encoderdecoder = self.encoderdecoder
        epochs = self.epochs
        loss_function = self.loss_function
        DEVICE = self.DEVICE
        optimizer = self.optimizer
        mean = []
        
        f= open(f"logs/results-{int(time.time())}.txt","w+")
        for epoch in range(epochs):
            print("Epoch: "+ str(epoch+1))
            training_indeces, testing_indeces = self.get_data_indeces()
            for j, training_index in enumerate(tqdm(training_indeces)):
                inputLEFT, inputRIGHT = self.get_input_arrays(training_index)
                encoderdecoder.zero_grad()
                output = encoderdecoder(inputLEFT.view(-1,7,256,640))
                loss = loss_function(output,[inputLEFT.view(-1,7,256,640), inputRIGHT.view(-1,3,256,640)])
                loss.backward()
                mean.append(loss.item())
                if j % 100 == 0:
                    trueloss = self.test_model(testing_indeces)
                    f.write(f"{round(sum(mean)/len(mean),5)}, {trueloss}\n")
                    f.flush()
                    mean = []
                    # Record the average training loss over time
                if j % 5000 == 0 and j != 0:
                    thetime = int(time.time())
                    torch.save(encoderdecoder.state_dict(), f"state_dicts/encoderdecoder-{thetime}")
                    # Save the weights to resume training from
                optimizer.step()
            thetime = int(time.time())
            torch.save(encoderdecoder.state_dict(), f"state_dicts/encoderdecoder-{thetime}_{epoch}")
        f.close()


def main():
    if len(sys.argv) > 1:
        network = trainer_tester(sys.argv[1])
    else:
        network = trainer_tester()
    #try:
    network.render_framerange(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    print('Framerange rendered')
    #except:
     #   print('~~~TRAINING NETWORK~~~')
      #  network.train()

if __name__ == "__main__":
    main()
