#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torchvision.utils as vutils
import torchvision.transforms as transforms
from misc import *
import models.DenseUnet as net
from PIL import Image
import torch.nn.functional as F
import numpy
import math
import time
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='datatest',  help='')
parser.add_argument('--dataroot', required=False,
  default='./dataset/rain/training/', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='./SOTS/hazy/', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=7, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=286, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=256, help='the height / width of the cropped input image to network')
parser.add_argument('--condition_GAN', type=int,
  default=1, help='set to 0 to use unconditional discriminator')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=48)
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lamdaA', type=float, default=0.0066, help='lambdaGAN')
parser.add_argument('--lamdaP', type=float, default=1, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='./DADN_ITS/netG_99.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=50, help='interval for evauating(generating) images from valDataroot')
parser.add_argument('--output', default='./compare_new/DADN_sots/', help="path to netG (to continue training)")
opt = parser.parse_args()
print(opt)
opt.workers=1
opt.manualSeed=1
# get dataloader
opt.dataset='datatest'
valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.imageSize, #opt.originalSize,
                          opt.imageSize,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0, 0, 0), std=(1, 1, 1),
                          split='endtoend',
                          shuffle=False,
                          seed=opt.manualSeed)

# get logger
trainLogger = open('%s/train.log' % opt.exp, 'w')


ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize
netG = net.G_HDC()

if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
#print(netG)

netG.cuda()



#def psnr(img1, img2):
#    mse = numpy.mean( (img1 - img2) ** 2 )
#    if mse == 0:
#        return 100
#    PIXEL_MAX = 255.0
#    # PIXEL_MAX = 1
#
#    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
import time
def padding_image(data):
    if data.size(2)%64==0 and data.size(3)%64==0:
        dim=(data.size(0),data.size(1),(int(data.size(2)/64))*64,(int(data.size(3)/64))*64)
    elif data.size(2)%64==0:
        dim=(data.size(0),data.size(1),(int(data.size(2)/64))*64,(int(data.size(3)/64)+1)*64)
    elif data.size(3)%64==0:
        dim=(data.size(0),data.size(1),(int(data.size(2)/64+1))*64,(int(data.size(3)/64))*64)
    else:
        dim=(data.size(0),data.size(1),(int(data.size(2)/64)+1)*64,(int(data.size(3)/64)+1)*64)
    size=numpy.array(data.size())
    padding = dim-size
    padding_h1= int(padding[3]/2)
    padding_h2=int(padding[3]-padding_h1)
    padding_w1=int(padding[2]/2)
    padding_w2=int(padding[2]-padding_w1)
    pad=(padding_h1,padding_h2,padding_w1,padding_w2)
    data=F.pad(data,pad,"constant",value=0)    
    return data,padding_w1,padding_h1,size
def bgrtorgb(dehaze):
    dehaze22=dehaze.clone()
    dehaze22[:,0,:,:] = dehaze[:,2,:,:]
    dehaze22[:,1,:,:] = dehaze[:,1,:,:]
    dehaze22[:,2,:,:] = dehaze[:,0,:,:]  
    return dehaze22
netG.eval()
index = 0
def let(m):
        transform=transforms.Compose([transforms.RandomVerticalFlip(p=1),
                                      transforms.RandomHorizontalFlip(p=1),
                                      transforms.ToTensor()
        ])
        m = m.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        m=Image.fromarray(m)
        m=transform(m).unsqueeze(0).cuda()
        return m
#to_pil_image = transforms.ToPILImage()
starttime=time.time()
for epoch in range(1):
  for i, data in enumerate(valDataloader, 0):
    netG.eval()
    name=data[1]
    newname=name[0].split('/')[-1]
    data=data[0]
    with torch.no_grad():
        starttime1=time.time()
        t0 = time.time()
        if data.size(2)%4==0 and data.size(3)%4==0:
            dim=(data.size(0),data.size(1),(int(data.size(2)/4))*4,(int(data.size(3)/4))*4)
        elif data.size(2)%4==0:
            dim=(data.size(0),data.size(1),(int(data.size(2)/4))*4,(int(data.size(3)/4)+1)*4)
        elif data.size(3)%4==0:
            dim=(data.size(0),data.size(1),(int(data.size(2)/4+1))*4,(int(data.size(3)/4))*4)
        else:
            dim=(data.size(0),data.size(1),(int(data.size(2)/4)+1)*4,(int(data.size(3)/4)+1)*4)
        size=numpy.array(data.size())
        padding = dim-size
        padding_h1= int(padding[3]/2)
        padding_h2=int(padding[3]-padding_h1)
        padding_w1=int(padding[2]/2)
        padding_w2=int(padding[2]-padding_w1)
        pad=(padding_h1,padding_h2,padding_w1,padding_w2)
        data=F.pad(data,pad,"constant",value=0)
        data = data.float().cuda()
#        dehaze=data.clone()
        dehaze = netG(data)
#        dehaze=netG(let(dehaze))
#        dehaze=netG(let(dehaze))
#        dehaze_out = dehaze[:,:,padding_w1:(size[2]+padding_w1),padding_h1:(size[3]+padding_h1)]
#        vutils.save_image(dehaze_out, opt.output+newname[:-4]+'_'+str(2+1)+'.png', normalize=False,scale_each=False) 
#        dehaze = netG(data)
        dehaze_out = dehaze[:,:,padding_w1:(size[2]+padding_w1),padding_h1:(size[3]+padding_h1)]
        vutils.save_image(dehaze_out, opt.output+newname, normalize=False,scale_each=False)
        endtime1=time.time()
        print('time:',endtime1-starttime1)
        index = index+1
        if opt.display == 1:
            dehaze = data.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            dehaze=Image.fromarray(dehaze)
            dehaze.show()
        print(index)
trainLogger.close()
endtime=time.time()
print(endtime-starttime)