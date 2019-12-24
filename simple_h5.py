#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
import torchvision.transforms as transforms
from misc import *
import models.DenseUnet as net

#from myutils.vgg16 import Vgg16
from myutils import utils
import pdb
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='testh5',  help='')
parser.add_argument('--dataroot', required=False,
  default='./facades/test_ihazy/', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='./facades/test256_new', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
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
parser.add_argument('--netG', default='./DUAN_256_baseline_vgg_gan/netG_149.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=50, help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
opt.workers=1

# get dataloader

dataloaderhaze = getLoader(opt.dataset,
                       opt.valDataroot,
                       opt.originalSize,
                       opt.imageSize,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
#                       mean=(0, 0, 0), std=(1, 1, 1),
                       split='train',
                       shuffle=False,
                       seed=opt.manualSeed)



ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

netG = net.G_HDC()
if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
print(netG)




netG.cuda()


import numpy
import math

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    # PIXEL_MAX = 1

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
import time

netG.eval()
index = 0
to_pil_image = transforms.ToPILImage()
t0 = time.time()
for epoch in range(1):
  for data_hazy,data_free in dataloaderhaze:
    with torch.no_grad():
        input_img= data_hazy
        input_img = input_img.float().cuda()     
        dehaze = netG(input_img) 
#        input_img = data_free.float().cuda()        
#        defree = netG(input_img) 
    directory='./result/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    vutils.save_image(dehaze, './compare_new/DADN_256_baseline_vgg_gan/'+str(index)+'.png', normalize=False,scale_each=False) 
#    vutils.save_image(data_free, './compare_new/gt/'+str(index)+'.png', normalize=False,scale_each=False)
#    vutils.save_image(data_hazy, './compare_new/hazy/'+str(index)+'.png', normalize=False,scale_each=False) 
#    vutils.save_image(defree, './compare_new/DADN_256_baseline_vgg_gan_gt/'+str(index)+'.png', normalize=False,scale_each=False)
    index = index+1
    print(index)
t1 = time.time()
print((t1-t0)/449)