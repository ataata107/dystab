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
from utils import save_ckp, load_ckp

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('ASPPConv') == -1 and classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        nn.init.normal_(m.weight.data, 0.0, math.sqrt(2. / n))
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def toInt3(elem):
    val  = elem.split("_")
    val = val[1].split('.')
    return int(val[0])

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 2

#Start Epoch
start_epoch = 0

# Number of training epochs
num_epochs = 30

# Learning rate for optimizers
lr = 0.0004

# Beta1 hyperparam for Adam optimizers
beta1 = 0.9
beta2 = 0.999

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#Rescale size
cnvrt_size = 256

#Checkpoint Path Inpainter
checkpoint_inpainter_path = "./checkpoint/inpainter/"

#Checkpoint Path Dynamic
checkpoint_dynamic_path = "./checkpoint/dynamic/"

pretrained_inpainter = None
pretrained_dynamic = None

if not os.path.exists(checkpoint_inpainter_path):
        os.makedirs(checkpoint_inpainter_path)
else:
  a1 = sorted(os.listdir(checkpoint_inpainter_path),key = toInt3,reverse= True)
  if(len(a1)>0):
    pretrained_inpainter = a1[0]

if not os.path.exists(checkpoint_dynamic_path):
        os.makedirs(checkpoint_dynamic_path)
else:
  a1 = sorted(os.listdir(checkpoint_dynamic_path),key = toInt3,reverse= True)
  if(len(a1)>0):
    pretrained_dynamic = a1[0]
flow_dataset = FlowDataset(transform = transforms.Compose([ToTensor(),Rescale((cnvrt_size,cnvrt_size))]))
# flow_dataset = FlowDataset(transform = transforms.Compose([ToTensor()]))

dataloader = DataLoader(flow_dataset, batch_size=batch_size,shuffle=True, num_workers=workers)

net_dynamic = createDeepLabv3().to(device)
net_dynamic.apply(weights_init)

net_impainter = Inpainter(ngpu=1).to(device) 
# net_impainter.apply(weights_init)
optimizerD = optim.Adam(net_dynamic.parameters(), lr=lr, betas=(beta1, beta2))
optimizerI = optim.Adam(net_impainter.parameters(), lr=lr, betas=(beta1, beta2))

if(pretrained_dynamic!=None):
  net_dynamic, optimizerD, start_epoch = load_ckp(checkpoint_dynamic_path+pretrained_dynamic, net_dynamic, optimizerD)
  print("Loaded pretrained: " + pretrained_dynamic)

if(pretrained_inpainter!=None):
  net_impainter, optimizerI, start_epoch = load_ckp(checkpoint_inpainter_path+pretrained_inpainter, net_impainter, optimizerI)
  print("Loaded pretrained: " + pretrained_inpainter)

loss_l1 = nn.L1Loss()
loss_l2 = nn.MSELoss()

I_losses = []
D_losses = []
iters = 0

print("Starting Training Loop... from" + str(start_epoch))
net_dynamic.train()
net_impainter.train()

step = 1
for epoch in range(start_epoch,num_epochs):
  step = 1
  for i, data in enumerate(dataloader, 0):
    # if(step%4!=0):
    net_dynamic.zero_grad()
    net_impainter.zero_grad()
    first_image = data['first_image'].to(device).float()

    flows = data['flows'].to(device)
    flow_forward = data['forward_flow'].to(device)
    flow_backward = data['backward_flow'].to(device)
    mask_flow_forward = net_dynamic(flow_forward)
    mask_flow_forward = mask_flow_forward['out']
    mask_flow_backward = net_dynamic(flow_backward)
    mask_flow_backward = mask_flow_backward['out']
    warped_flow_backward,mask1 = warp(mask_flow_backward,flow_forward)
    t_c_loss_1 = loss_l1(mask1*mask_flow_forward, warped_flow_backward*mask1)
    warped_flow_forward,mask2 = warp(mask_flow_forward,flow_backward)
    t_c_loss_2 = loss_l1(mask2*mask_flow_backward, warped_flow_forward*mask2)

    # print(flows.shape,first_image.shape,first_image.dtype)
    err_num1 = loss_l2(torch.ones(5).to(device), torch.ones(5).to(device)) #0
    err_num2 = loss_l2(torch.ones(5).to(device), torch.ones(5).to(device)) #0
    err_den1 = loss_l2(torch.ones(5).to(device), torch.ones(5).to(device)) #0
    err_den2 = loss_l2(torch.ones(5).to(device), torch.ones(5).to(device)) #0
    for k,flow in enumerate(torch.split(flows,1,1)):
      flow = torch.squeeze(flow, 1)
      # print(flow.shape)
      mask = net_dynamic(flow)
      mask = mask['out']
      inpainted_1 = net_impainter(first_image,(1-mask)*flow,mask,(256,256)) 
      inpainted_2 = net_impainter(first_image,(mask)*flow,1-mask,(256,256)) 
      # err_num1+=loss_l2(mask*flow,inpainted_1)
      # err_num2+=loss_l2((1-mask)*flow,inpainted_2)
      err_num1+= loss_l2(mask*flow, inpainted_1) #torch.norm(mask*flow-inpainted_1)
      err_num2+= loss_l2((1-mask)*flow, inpainted_2) #torch.norm((1-mask)*flow-inpainted_2)
      err_den1+= loss_l2(mask*flow/2, -mask*flow/2) #torch.norm(mask*flow)
      err_den2+= loss_l2((1-mask)*flow/2, -(1-mask)*flow/2) #torch.norm((1-mask)*flow)
      # print(flow.shape, mask.shape,inpainted.shape)
    err = (err_num1/(err_den1+0.0001))+(err_num2/(err_den2+0.0001))
    # err.backward()
    if(step%4!=0):
      err = -err
      err.backward()
      t_c_loss = (t_c_loss_1+t_c_loss_2)
      t_c_loss.backward()
      optimizerD.step()
    else:
      err.backward()
      optimizerI.step()
    print("Epoch"+str(epoch),"Step"+str(step),abs(err.item()),abs(t_c_loss.item()))
    if(step%200==0):
      checkpoint_dynamic = {
          'epoch': epoch + 1,
          'state_dict': net_dynamic.state_dict(),
          'optimizer': optimizerD.state_dict(),
      }

      checkpoint_inpainter = {
          'epoch': epoch + 1,
          'state_dict': net_impainter.state_dict(),
          'optimizer': optimizerI.state_dict(),
      }

      save_ckp(checkpoint_dynamic, checkpoint_dynamic_path+"checkpoint_"+str(epoch+1)+".pt")
      save_ckp(checkpoint_inpainter, checkpoint_inpainter_path+"checkpoint_"+str(epoch+1)+".pt")
 

    step+=1
    
    # break
  checkpoint_dynamic = {
      'epoch': epoch + 1,
      'state_dict': net_dynamic.state_dict(),
      'optimizer': optimizerD.state_dict(),
  }

  checkpoint_inpainter = {
      'epoch': epoch + 1,
      'state_dict': net_impainter.state_dict(),
      'optimizer': optimizerI.state_dict(),
  }

  save_ckp(checkpoint_dynamic, checkpoint_dynamic_path+"checkpoint_"+str(epoch+1)+".pt")
  save_ckp(checkpoint_inpainter, checkpoint_inpainter_path+"checkpoint_"+str(epoch+1)+".pt")

  # break
