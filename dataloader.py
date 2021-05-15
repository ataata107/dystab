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

def toInt1(elem):
    
    return int(elem)

def toInt2(elem):
    val  = elem.split("_")
    val = val[1].split('.')
    return int(val[0])

class FlowDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_pairs = sorted(os.listdir("./Dataset/image_pair"),key = toInt1)
        self.flow_pairs = sorted(os.listdir("./Dataset/flow_pair"),key = toInt1)
        self.flows = sorted(os.listdir("./Dataset/flows"),key = toInt1)
        print("Done")
        # self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir = os.path.join("./Dataset/image_pair",
                                self.image_pairs[idx])
        imgs = sorted(os.listdir(img_dir),key = toInt2)
        first_image_dir = os.path.join(img_dir,imgs[0])
        second_image_dir = os.path.join(img_dir,imgs[1])
        first_image = io.imread(first_image_dir)
        second_image = io.imread(second_image_dir)

        flow_pair_dir = os.path.join("./Dataset/flow_pair",
                                self.flow_pairs[idx])
        forward_flow_dir = os.path.join(flow_pair_dir,'forward.npy')
        backward_flow_dir = os.path.join(flow_pair_dir,'backward.npy')
        forward_flow = np.load(forward_flow_dir)
        backward_flow = np.load(backward_flow_dir)

        flows_dir = os.path.join("./Dataset/flows",
                                self.flows[idx])
        flows_dir_main = sorted(os.listdir(flows_dir))
        flows = []
        for i in flows_dir_main:
          dir_flow = os.path.join(flows_dir,i)
          curr_flow = np.load(dir_flow)
          flows.append(curr_flow)
        flows = np.array(flows)
        # print(flows_dir)
        # print(flows.shape)
        # print(first_image_dir)
        # print(first_image.shape)
        # print(second_image_dir)
        # print(second_image.shape)
        # print(forward_flow_dir)
        # print(forward_flow.shape)
        # print(backward_flow_dir)
        # print(backward_flow.shape)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'first_image': first_image, 'second_image' : second_image, 'forward_flow': forward_flow,'backward_flow': backward_flow, 'flows': flows}

        if self.transform:
          sample = self.transform(sample)
        # print(sample['first_image'].shape)
        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        # assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        first_image, second_image, forward_flow, backward_flow,flows = sample['first_image'], sample['second_image'], sample['forward_flow'], sample['backward_flow'], sample['flows']
        h_f, w_f = first_image.shape[:2]
        h_s, w_s = second_image.shape[:2]
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        new_h, new_w = self.output_size
        # new_h_s, new_w_s = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        # new_h_s, new_w_s = int(new_h_s), int(new_w_s)

        first_image = transforms.Resize((new_h, new_w))(first_image)
        second_image = transforms.Resize((new_h, new_w))(second_image)
        forward_flow = transforms.Resize((new_h, new_w))(forward_flow)
        backward_flow = transforms.Resize((new_h, new_w))(backward_flow)
        flows = transforms.Resize((new_h, new_w))(flows)
        # second_image = transform.resize(second_image, (new_h, new_w))
        # forward_flow = transform.resize(forward_flow, (new_h, new_w))
        # backward_flow = transform.resize(backward_flow, (new_h, new_w))
        #flows = transform.resize(flows, (new_h, new_w))
        # print(img.shape)
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * [new_w / w, new_h / h]

        return {'first_image': first_image, 'second_image' : second_image, 'forward_flow': forward_flow, 'backward_flow': backward_flow, 'flows': flows}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        first_image, second_image, forward_flow, backward_flow,flows = sample['first_image'], sample['second_image'], sample['forward_flow'], sample['backward_flow'], sample['flows']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        first_image = first_image.transpose((2, 0, 1))
        second_image = second_image.transpose((2, 0, 1))
        forward_flow = forward_flow.transpose((2, 0, 1))
        backward_flow = backward_flow.transpose((2, 0, 1))
        flows  = flows.transpose((0,3,1,2))
        # print(flows.shape)
        # flows = transform.resize(flows, (self.new_h, self.new_w))
        
        return {'first_image': torch.from_numpy(first_image), 'second_image' : torch.from_numpy(second_image), 'forward_flow' : torch.from_numpy(forward_flow), 'backward_flow' : torch.from_numpy(backward_flow), 'flows': torch.from_numpy(flows)}
