from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import math
import random
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataloader import FlowDataset, Rescale, ToTensor
from utils import warp
from model import createDeepLabv3, Inpainter
from pad import pad

def createDeepLabv3(outputchannels=1):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True)
    model.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    # model.train()
    return model


        # iters += 1




class Inpainter(nn.Module):
    def __init__(self, ngpu,f=1.0):
        super(Inpainter, self).__init__()
        self.ngpu = ngpu
        self.C =2
        self.f = int(f)
        self.conv1 = nn.Conv2d(3, 64*self.f, 7, 2, bias=True)
        self.conv2 = nn.Conv2d(64*self.f, 128*self.f, 5, 2, bias=True)
        self.conv3 = nn.Conv2d(128*self.f, 256*self.f, 5, 2, bias=True)
        self.conv4 = nn.Conv2d(256*self.f, 256*self.f, 3, 1, bias=True)
        self.conv5 = nn.Conv2d(256*self.f, 512*self.f, 3, 2, bias=True)
        self.conv6 = nn.Conv2d(512*self.f, 512*self.f, 3, 1, bias=True)
        self.conv7 = nn.Conv2d(512*self.f, 512*self.f, 3, 2, bias=True)
        self.conv8 = nn.Conv2d(512*self.f, 512*self.f, 3, 1, bias=True)
        self.conv9 = nn.Conv2d(512*self.f, 512*self.f, 3, 2, bias=True)

        self.conv10 = nn.Conv2d(3, 64*self.f, 7, 2, bias=True)
        self.conv11 = nn.Conv2d(64*self.f, 128*self.f, 5, 2, bias=True)
        self.conv12 = nn.Conv2d(128*self.f, 256*self.f, 5, 2, bias=True)
        self.conv13 = nn.Conv2d(256*self.f, 256*self.f, 3, 1, bias=True)
        self.conv14 = nn.Conv2d(256*self.f, 512*self.f, 3, 2, bias=True)
        self.conv15 = nn.Conv2d(512*self.f, 512*self.f, 3, 1, bias=True)
        self.conv16 = nn.Conv2d(512*self.f, 512*self.f, 3, 2, bias=True)
        self.conv17 = nn.Conv2d(512*self.f, 512*self.f, 3, 1, bias=True)
        self.conv18 = nn.Conv2d(512*self.f, 512*self.f, 3, 2, bias=True)

        self.conv19 = nn.Conv2d(512*2*self.f, 512*self.f, 4, 1, bias=True)
        self.conv20 = nn.Conv2d(512*2*self.f, self.C, 3, 1, bias=True)
        self.conv21 = nn.Conv2d(512*2*self.f, 512*self.f, 4, 1, bias=True)
        self.conv22 = nn.Conv2d(self.C, self.C, 4, 1, bias=True)
        self.conv23 = nn.Conv2d(512*2*self.f+self.C, self.C, 3, 1, bias=True)
        self.conv24 = nn.Conv2d(512*2*self.f+self.C, 256*self.f, 4, 1, bias=True)
        self.conv25 = nn.Conv2d(self.C, self.C, 4, 1, bias=True)
        self.conv26 = nn.Conv2d(256*2*self.f+self.C, self.C, 3, 1, bias=True)
        self.conv27 = nn.Conv2d(256*2*self.f+self.C, 128*self.f, 4, 1, bias=True)
        self.conv28 = nn.Conv2d(self.C, self.C, 4, 1, bias=True)
        self.conv29 = nn.Conv2d(128*2*self.f+self.C, self.C, 3, 1, bias=True)
        self.conv30 = nn.Conv2d(128*2*self.f+self.C, 64*self.f, 4, 1, bias=True)
        self.conv31 = nn.Conv2d(self.C, self.C, 4, 1, bias=True)
        self.conv32 = nn.Conv2d(64*2*self.f+self.C, self.C, 5, 1, bias=True)
        # self.main = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     nn.ReLU(True),
        #     # state size. (ngf*8) x 4 x 4
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.ReLU(True),
        #     # state size. (ngf*4) x 8 x 8
        #     nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.ReLU(True),
        #     # state size. (ngf*2) x 16 x 16
        #     nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     # state size. (ngf) x 32 x 32
        #     nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
        #     nn.Tanh()
        #     # state size. (nc) x 64 x 64

        #     nn.Conv2d(2, 64*self.f, 7, 2, 1, bias=True),
        # )

    def forward(self, img,flow,mask,orisize):
        ## Confirm padding, make correct x
        #flow = concat backgound and foreground
        x = torch.cat((mask,flow),1)
        x = pad(x,size=(x.shape[2],x.shape[3]),kernel_size=7,stride=2,dilation=1)
        x = self.conv1(x)
        x = nn.LeakyReLU(0.3)(x)

        x = pad(x,size=(x.shape[2],x.shape[3]),kernel_size=5,stride=2,dilation=1)
        x = self.conv2(x)
        x = nn.LeakyReLU(0.3)(x)

        x = pad(x,size=(x.shape[2],x.shape[3]),kernel_size=5,stride=2,dilation=1)
        x = self.conv3(x)
        x = nn.LeakyReLU(0.3)(x)

        x = pad(x,size=(x.shape[2],x.shape[3]),kernel_size=3,stride=1,dilation=1)
        x = self.conv4(x)
        x = nn.LeakyReLU(0.3)(x)

        x = pad(x,size=(x.shape[2],x.shape[3]),kernel_size=3,stride=2,dilation=1)
        x = self.conv5(x)
        x = nn.LeakyReLU(0.3)(x)

        x = pad(x,size=(x.shape[2],x.shape[3]),kernel_size=3,stride=1,dilation=1)
        x = self.conv6(x)
        x = nn.LeakyReLU(0.3)(x)

        x = pad(x,size=(x.shape[2],x.shape[3]),kernel_size=3,stride=2,dilation=1)
        x = self.conv7(x)
        x = nn.LeakyReLU(0.3)(x)

        x = pad(x,size=(x.shape[2],x.shape[3]),kernel_size=3,stride=1,dilation=1)
        x = self.conv8(x)
        x = nn.LeakyReLU(0.3)(x)

        x = pad(x,size=(x.shape[2],x.shape[3]),kernel_size=3,stride=2,dilation=1)
        x = self.conv9(x)
        x = nn.LeakyReLU(0.3)(x)
        ##############################################################################

        bconv1 = pad(img,size=(img.shape[2],img.shape[3]),kernel_size=7,stride=2,dilation=1)
        bconv1 = self.conv10(bconv1)
        bconv1 = nn.LeakyReLU(0.3)(bconv1)

        bconv2 = pad(bconv1,size=(bconv1.shape[2],bconv1.shape[3]),kernel_size=5,stride=2,dilation=1)
        bconv2 = self.conv11(bconv2)
        bconv2 = nn.LeakyReLU(0.3)(bconv2)

        bconv3 = pad(bconv2,size=(bconv2.shape[2],bconv2.shape[3]),kernel_size=5,stride=2,dilation=1)
        bconv3 = self.conv12(bconv3)
        bconv3 = nn.LeakyReLU(0.3)(bconv3)

        bconv31 = pad(bconv3,size=(bconv3.shape[2],bconv3.shape[3]),kernel_size=3,stride=1,dilation=1)
        bconv31 = self.conv13(bconv31)
        bconv31 = nn.LeakyReLU(0.3)(bconv31)

        bconv4 = pad(bconv31,size=(bconv31.shape[2],bconv31.shape[3]),kernel_size=3,stride=2,dilation=1)
        bconv4 = self.conv14(bconv4)
        bconv4 = nn.LeakyReLU(0.3)(bconv4)

        bconv41 = pad(bconv4,size=(bconv4.shape[2],bconv4.shape[3]),kernel_size=3,stride=1,dilation=1)
        bconv41 = self.conv15(bconv41)
        bconv41 = nn.LeakyReLU(0.3)(bconv41)

        bconv5 = pad(bconv41,size=(bconv41.shape[2],bconv41.shape[3]),kernel_size=3,stride=2,dilation=1)
        bconv5 = self.conv16(bconv5)
        bconv5 = nn.LeakyReLU(0.3)(bconv5)

        bconv51 = pad(bconv5,size=(bconv5.shape[2],bconv5.shape[3]),kernel_size=3,stride=1,dilation=1)
        bconv51 = self.conv17(bconv51)
        bconv51 = nn.LeakyReLU(0.3)(bconv51)

        bconv6 = pad(bconv51,size=(bconv51.shape[2],bconv51.shape[3]),kernel_size=3,stride=2,dilation=1)
        bconv6 = self.conv18(bconv6)
        bconv6 = nn.LeakyReLU(0.3)(bconv6)
        ###############################################################################

        conv6 = torch.cat((x,bconv6),1)
        deconv5 = torch.nn.Upsample((bconv51.shape[2],bconv51.shape[3]))(conv6)
        deconv5 = pad(deconv5,size=(deconv5.shape[2],deconv5.shape[3]),kernel_size=4,stride=1,dilation=1)
        deconv5 = self.conv19(deconv5)
        deconv5 = nn.LeakyReLU(0.3)(deconv5)
        concat5 = torch.cat((deconv5,bconv51),1)

        flow5 = pad(concat5,size=(concat5.shape[2],concat5.shape[3]),kernel_size=3,stride=1,dilation=1)
        flow5 = self.conv20(flow5)
        m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        flow5=m(flow5)

        deconv4 = torch.nn.Upsample((bconv41.shape[2],bconv41.shape[3]))(concat5)
        deconv4 = pad(deconv4,size=(deconv4.shape[2],deconv4.shape[3]),kernel_size=4,stride=1,dilation=1)
        deconv4 = self.conv21(deconv4)
        deconv4 = nn.LeakyReLU(0.3)(deconv4)
        upflow4 = torch.nn.Upsample((bconv41.shape[2],bconv41.shape[3]))(flow5)
        upflow4 = pad(upflow4,size=(upflow4.shape[2],upflow4.shape[3]),kernel_size=4,stride=1,dilation=1)
        upflow4 = self.conv22(upflow4)
        m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        upflow4=m(upflow4)
        concat4 = torch.cat((deconv4,bconv41,upflow4),1)

        flow4 = pad(concat4,size=(concat4.shape[2],concat4.shape[3]),kernel_size=3,stride=1,dilation=1)
        flow4 = self.conv23(flow4)
        m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        flow4=m(flow4)

        deconv3 = torch.nn.Upsample((bconv31.shape[2],bconv31.shape[3]))(concat4)
        deconv3 = pad(deconv3,size=(deconv3.shape[2],deconv3.shape[3]),kernel_size=4,stride=1,dilation=1)
        deconv3 = self.conv24(deconv3)
        deconv3 = nn.LeakyReLU(0.3)(deconv3)
        upflow3 = torch.nn.Upsample((bconv31.shape[2],bconv31.shape[3]))(flow4)
        upflow3 = pad(upflow3,size=(upflow3.shape[2],upflow3.shape[3]),kernel_size=4,stride=1,dilation=1)
        upflow3 = self.conv25(upflow3)
        m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        upflow3=m(upflow3)
        concat3 = torch.cat((deconv3,bconv31,upflow3),1)

        flow3 = pad(concat3,size=(concat3.shape[2],concat3.shape[3]),kernel_size=3,stride=1,dilation=1)
        flow3 = self.conv26(flow3)
        m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        flow3=m(flow3)

        deconv2 = torch.nn.Upsample((bconv2.shape[2],bconv2.shape[3]))(concat3)
        deconv2 = pad(deconv2,size=(deconv2.shape[2],deconv2.shape[3]),kernel_size=4,stride=1,dilation=1)
        deconv2 = self.conv27(deconv2)
        deconv2 = nn.LeakyReLU(0.3)(deconv2)
        upflow2 = torch.nn.Upsample((bconv2.shape[2],bconv2.shape[3]))(flow3)
        upflow2 = pad(upflow2,size=(upflow2.shape[2],upflow2.shape[3]),kernel_size=4,stride=1,dilation=1)
        upflow2 = self.conv28(upflow2)
        m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        upflow2=m(upflow2)
        concat2 = torch.cat((deconv2,bconv2,upflow2),1)

        flow2 = pad(concat2,size=(concat2.shape[2],concat2.shape[3]),kernel_size=3,stride=1,dilation=1)
        flow2 = self.conv29(flow2)
        m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        flow2=m(flow2)

        deconv1 = torch.nn.Upsample((bconv1.shape[2],bconv1.shape[3]))(concat2)
        deconv1 = pad(deconv1,size=(deconv1.shape[2],deconv1.shape[3]),kernel_size=4,stride=1,dilation=1)
        deconv1 = self.conv30(deconv1)
        deconv1 = nn.LeakyReLU(0.3)(deconv1)
        upflow1 = torch.nn.Upsample((bconv1.shape[2],bconv1.shape[3]))(flow2)
        upflow1 = pad(upflow1,size=(upflow1.shape[2],upflow1.shape[3]),kernel_size=4,stride=1,dilation=1)
        upflow1 = self.conv31(upflow1)
        m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        upflow1=m(upflow1)
        concat1 = torch.cat((deconv1,bconv1,upflow1),1)

        flow1 = pad(concat1,size=(concat1.shape[2],concat1.shape[3]),kernel_size=5,stride=1,dilation=1)
        flow1 = self.conv32(flow1)
        m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        flow1=m(flow1)

        flow0 = torch.nn.Upsample((orisize[0],orisize[1]))(flow1)

        return flow0

