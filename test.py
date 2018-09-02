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

from misc import *
import models.dehaze22  as net

from myutils.vgg16 import Vgg16
from myutils import utils
import pdb
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='pix2pix',  help='')
parser.add_argument('--dataroot', required=False,
  default='./dataset/rain/training/', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='./dataset/rain/test/', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=7, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=10, help='input batch size')
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
parser.add_argument('--netG', default='./models/netG_epoch.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=50, help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)
opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
opt.workers=1

# get dataloader
valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.imageSize, #opt.originalSize,
                          opt.imageSize,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='Train',
                          shuffle=False,
                          seed=opt.manualSeed)

# get logger
trainLogger = open('%s/train.log' % opt.exp, 'w')


ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize


netG = net.G(inputChannelSize, outputChannelSize, ngf)


if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
print(netG)



#inputs have two img in a jpg
val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_target, val_input = val_target.cuda(), val_input.cuda()
val_iter = iter(valDataloader)
data_val = val_iter.next()
val_input_cpu, val_target_cpu = data_val
val_input_cpu, val_target_cpu = val_input_cpu.float().cuda(), val_target_cpu.float().cuda()
val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)


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


# NOTE training loop
ganIterations = 0
index=0
psnrall = 0
ssimall=0
iteration = 0
# print(1)
for epoch in range(1):
  for i, data in enumerate(valDataloader, 0):
    t0 = time.time()
    input_img, input_img2= data
    input_img = input_img.float().cuda()    
    dehaze = netG(input_img)
    index = 0
    directory='./result/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    vutils.save_image(dehaze, './result/'+str(index)+'_IDCGAN_derain.png', normalize=True, range=(-1, 1),scale_each=False)
    index = index+1
    print(index)
trainLogger.close()