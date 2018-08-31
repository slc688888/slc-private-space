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
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=50, help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)
opt.manualSeed = random.randint(1, 10000)
# opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
opt.workers=1


# get dataloader
dataloader = getLoader(opt.dataset,
                       opt.dataroot,
                       opt.originalSize,
                       opt.imageSize,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='train',
                       shuffle=True,
                       seed=opt.manualSeed)

opt.dataset='pix2pix_val2'
valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.imageSize, #opt.originalSize,
                          opt.imageSize,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='val',
                          shuffle=False,
                          seed=opt.manualSeed)
# get logger
trainLogger = open('%s/train.log' % opt.exp, 'w')

#train_list=glob.glob(opt.dataroot+'/*.jpg')
#print(len(train_list))
#取得重要参数
real_label = 1
fake_label = 0
ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize
idx_A = {inputChannelSize+1, inputChannelSize+outputChannelSize}
idx_B = {1, inputChannelSize}

#建立生成网络netG和判别网络netD

netG = net.G(inputChannelSize, outputChannelSize, ngf)
netG.apply(weights_init)
if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = net.D(inputChannelSize,outputChannelSize, ndf,3)
netD.apply(weights_init)
if opt.netD != '':
  netD.load_state_dict(torch.load(opt.netD))
print(netD)

netG.train()
netD.train()


#设置损失函数loss function

criterionBCE = nn.BCELoss()
criterionMSE = nn.MSELoss()

#留出空间
real_A = torch.Tensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
real_B = torch.Tensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
fake_B = torch.Tensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
real_AB = torch.Tensor(opt.batchSize, outputChannelSize + inputChannelSize*opt.condition_GAN, opt.imageSize, opt.imageSize)
fake_AB = torch.Tensor(opt.batchSize, outputChannelSize + inputChannelSize*opt.condition_GAN, opt.imageSize, opt.imageSize)

errD, errG, errL1 = 0, 0, 0

real_A = real_A.cuda()
real_B = real_B.cuda()
fake_B = real_B.cuda()
real_AB = real_AB.cuda()
fake_AB = real_AB.cuda()
netD.cuda()
netG.cuda()
criterionBCE.cuda()
criterionMSE.cuda()
print('done')

lamdaA = opt.lamdaA
lamdaP = opt.lamdaP

# Initialize VGG-16
vgg = Vgg16()
utils.init_vgg16('./models/')
vgg.load_state_dict(torch.load(os.path.join('./models/', "vgg16.weight")))
vgg.cuda()



# pdb.set_trace()
# get optimizer 设置优化器
optimizerD = optim.Adam(netD.parameters(), lr = opt.lrD, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999))


#测试用实际数据
val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_target, val_input = val_target.cuda(), val_input.cuda()
val_iter = iter(valDataloader)
data_val = val_iter.next()
val_input_cpu, val_target_cpu = data_val
val_input_cpu, val_target_cpu = val_input_cpu.float().cuda(), val_target_cpu.float().cuda()
val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)

# NOTE training loop
ganIterations = 0
for epoch in range(opt.niter):
    if epoch > opt.annealStart:
        adjust_learning_rate(optimizerD, opt.lrD, epoch, None, opt.annealEvery)
        adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)
    for i, data in enumerate(dataloader, 0):         
        #get img
        real_A, real_B= data
        real_A, real_B = real_A.float().cuda(), real_B.float().cuda()
        real_AB = torch.cat((real_A,real_B),1)    
        fake_B = netG(real_A)
        fake_AB = torch.cat((real_A,fake_B),1)
            
         #netD
        for p in netD.parameters():
            p.requires_grad = True
        netD.zero_grad()
        output = netD(real_AB)
        label = torch.FloatTensor(output.size()).fill_(real_label).cuda()
        errD_real = criterionBCE(output,label)
        errD_real.backward()
            
        output = netD(fake_AB)
        label.fill_(fake_label)
        errD_fake = criterionBCE(output,label)
        errD_fake.backward()
            
        errD = (errD_real + errD_fake)/2
        optimizerD.step()
        
        #netG
        netG.zero_grad()
        #GAN loss La
        df_dg = torch.zeros(fake_B.size()).cuda()
        output = netD(real_AB)
        label = torch.FloatTensor(output.size()).fill_(real_label).cuda()
        errLa = criterionBCE(output, label)
        #Eucledean loss Le
        df_do_AE = torch.zeros(fake_B.size()).cuda()
        fake_B = netG(real_A)
        errLe = criterionMSE(fake_B, real_B)
        #Perceptual loss Lp
        df_do_AE1 = torch.zeros(fake_B.size()).cuda()
        features_content = vgg(fake_B)
        f_xc_c = Variable(features_content[1].data, requires_grad=False)
        features_y = vgg(real_B)
        errLp =  criterionMSE(features_y[1], f_xc_c)
        
        errG = errLe + errLa*lamdaA + errLp*lamdaP
        errG.backward(retain_graph =True)
        optimizerG.step()

        ganIterations += 1
    
        if ganIterations % opt.display == 0:
            print('[%d/%d][%d/%d] L_D: %f L_G: %f La: %f Lp: %f Le: %f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], errLa.data[0], errLp.data[0], errLe.data[0]))
            sys.stdout.flush()
            trainLogger.write('%d\t%f\t%f\t%f\t%f\t%f\n' % \
                            (i, errD.data[0], errG.data[0], errLa.data[0], errLp.data[0], errLe.data[0]))
            trainLogger.flush()
    
        if epoch % 2 == 0:            
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.exp, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.exp, epoch))
        if ganIterations % opt.evalIter == 0:  
            val_batch_output = torch.FloatTensor(val_input.size()).fill_(0)
            for idx in range(val_input.size(0)):
                single_img = val_input[idx,:,:,:].unsqueeze(0)
                val_inputv = Variable(single_img, volatile=True)
                dehaze21 = netG(val_inputv)
                val_batch_output[idx,:,:,:].copy_(dehaze21[0,:,:,:])
                del dehaze21
                del single_img
                del val_inputv
            vutils.save_image(val_batch_output, '%s/generated_epoch_%08d_iter%08d.png' % \
                    (opt.exp, epoch, ganIterations), normalize=False, scale_each=False)
            del val_batch_output
        
          
trainLogger.close()
