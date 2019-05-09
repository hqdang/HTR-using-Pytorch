from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from DL import DataLoader, Batch
import decode_beam
import tensorflow as tf
from SamplePreprocessor import preprocess
import os
import pathlib
import BestPath
import Common
import time
import copy
import math

class Model(torch.nn.Module): 
        
    # model constants
    batchSize = 50
    imgSize = (128, 32)
    maxTextLen = 32
    
    def __init__(self):
        super(Model, self).__init__()
        self.snapID = 0
        #self.is_train = tf.placeholder(tf.bool, name='is_train')
        
        # input image batch
        #self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))
        #self.inputImgs = torch.tensor(batch.imgs, Model.imgSize[0], Model.imgSize[1])
        
        #cnn function
        self.conv1 = torch.nn.Conv2d(1, 32, 5, stride = 1, padding = 2).cuda()
        self.batchnorm1 = torch.nn.BatchNorm2d(32).cuda().cuda()
        self.relu1 = torch.nn.ReLU().cuda()
        self.pool1 = torch.nn.MaxPool2d((2,2), stride = (2,2)).cuda()
        
        self.conv2 = torch.nn.Conv2d(32, 64, 5, stride = 1, padding = 2).cuda()
        self.batchnorm2 = torch.nn.BatchNorm2d(64).cuda()
        self.relu2 = torch.nn.ReLU().cuda()
        self.pool2 = torch.nn.MaxPool2d((2,2), stride = (2,2)).cuda()
        
        self.conv3 = torch.nn.Conv2d(64, 128, 3, stride = 1, padding = 1).cuda()
        self.batchnorm3 = torch.nn.BatchNorm2d(128).cuda()
        self.relu3 = torch.nn.ReLU().cuda()
        self.pool3 = torch.nn.MaxPool2d((1,2), stride = (1,2)).cuda()  
        
        self.conv4 = torch.nn.Conv2d(128, 128, 3, stride = 1, padding = 1).cuda()
        self.batchnorm4 = torch.nn.BatchNorm2d(128).cuda()
        self.relu4 = torch.nn.ReLU().cuda()
        self.pool4 = torch.nn.MaxPool2d((1,2), stride = (1,2)).cuda()
        
        self.conv5 = torch.nn.Conv2d(128, 256, 3, stride = 1, padding = 1).cuda()
        self.batchnorm5 = torch.nn.BatchNorm2d(256).cuda()
        self.relu5 = torch.nn.ReLU().cuda()
        self.pool5 = torch.nn.MaxPool2d((1,2), stride = (1,2)).cuda()
        
        #rnn function
        self.lstm = torch.nn.LSTM(256, 256, 2, batch_first = True, bidirectional = True).cuda()
        # BxTxH + BxTxH -> BxTx2H -> BxTx1x2H 
        #squeeze
        self.rnnconv2d = torch.nn.Conv2d(512, 80, 1, stride = 1, padding = 0).cuda()
        
    def forward(self, inputImgs):
        #cnn forward pass
        inputTensor = torch.from_numpy(inputImgs).cuda()
        inputTensor = inputTensor.type(torch.FloatTensor).cuda()
        inputTensor = torch.unsqueeze(inputTensor, 1)
        #print (inputTensor.size()) [50,1,128,32]
        out = self.conv1(inputTensor)
        out = self.batchnorm1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = self.conv4(out)
        out = self.batchnorm4(out)
        out = self.relu4(out)
        out = self.pool4(out)
        out = self.conv5(out)
        out = self.batchnorm5(out)
        out = self.relu5(out)
        out = self.pool5(out)
        
        #rnn forward pass
        #print (cnn.size())
        out = torch.squeeze(out, 3)
        out = out.permute(0,2,1)
        #print (cnn.size()) cnn= [50,32,256]
        #h0, c0 shape (num_layers * num_directions, batch, hidden_size):
        h0 = torch.zeros(4, out.size(0), 256).cuda()
        c0 = torch.zeros(4, out.size(0), 256).cuda()
        #packed_cnn = torch.nn.utils.rnn.pack_padded_sequence(cnn, [32]*cnn.size(1))
        #print(packed_cnn.size())
        out, _ = self.lstm(out, (h0,c0))
        #print (rnn_out.size())
        #rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=False)
        #print (rnn_out.size()) [50,32,512]
        out = torch.unsqueeze(out, 3) #[50,512,32,1]
        out = out.permute(0,2,1,3) #[50,32,1,512]
        #print (rnn_out.size())
        #print (rnn_out.size())
        out = self.rnnconv2d(out)
        #out = self.fnl(rnn_out)
        return out