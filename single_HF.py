#!/usr/bin/env python3
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
import models.U_net as net
import itertools
from myutils.vgg16 import Vgg16
from myutils import utils



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='haze',  help='')
parser.add_argument('--dataroot', required=False,
  default='./image/haze/', help='path to trn dataset')
parser.add_argument('--datarootfree', required=False,
  default='./image/free/', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='./test/test/', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=4, help='input batch size')
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
parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.002, help='learning rate, default=0.0002')
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
parser.add_argument('--exp', default='sampleHFA', help='folder to output images and model checkpoints')
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
dataloaderhaze = getLoader(opt.dataset,
                       opt.dataroot,
                       opt.originalSize,
                       opt.imageSize,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='train',
                       shuffle=True,
                       seed=opt.manualSeed)
opt.dataset='gt'
dataloaderfree = getLoader(opt.dataset,
                       opt.datarootfree,
                       opt.originalSize,
                       opt.imageSize,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='train',
                       shuffle=True,
                       seed=opt.manualSeed)
opt.dataset = 'test'
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
haze_label = 1
free_label = 0
ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize
idx_A = {inputChannelSize+1, inputChannelSize+outputChannelSize}
idx_B = {1, inputChannelSize}

#建立生成网络netG和判别网络netD

netG = net.G(inputChannelSize, outputChannelSize)
#netG_B = net.G(inputChannelSize, outputChannelSize)
netG.apply(weights_init)
#netG_B.apply(weights_init)
#if opt.netG != '':
#  netG.load_state_dict(torch.load(opt.netG))
print(netG)
#print(netG_B)

netD_B = net.D(inputChannelSize,outputChannelSize, ndf,3)
netD_A = net.D(inputChannelSize,outputChannelSize, ndf,3)
netD_A.apply(weights_init)
netD_B.apply(weights_init)
if opt.netD != '':
  netD_A.load_state_dict(torch.load(opt.netD_A))
print(netD_A)

#netG.train()
#netD.train()


#设置损失函数loss function

criterionBCE = nn.BCELoss()
criterionMSE = nn.MSELoss()
criterionCycle = nn.L1Loss()
#留出空间
#haze_train = torch.Tensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
#free_train = torch.Tensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)

errD, errG, errL1 = 0, 0, 0

#haze_train = haze_train.cuda()
#free_train = free_train.cuda()
netD_A.cuda()
netD_B.cuda()
netG.cuda()
criterionBCE.cuda()
criterionMSE.cuda()
criterionCycle.cuda()
print('done')

lamdaA = opt.lamdaA
lamdaP = opt.lamdaP

# Initialize VGG-16
vgg = Vgg16()
utils.init_vgg16('./models/')
vgg.load_state_dict(torch.load(os.path.join('./models/', "vgg16.weight")))
vgg.cuda()



# pdb.set_trace()
''' set optimizer'''
optimizerD = optim.Adam(netD_A.parameters(), lr = opt.lrD, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999))


'''test data'''
val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_target, val_input = val_target.cuda(), val_input.cuda()
val_iter = iter(valDataloader)
data_val = val_iter.next()
val_input_cpu= data_val
val_input_cpu= val_input_cpu.float().cuda()
val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)

#def set_requires_grad(self, netts, requires_grad=False):
#    if not isinstance(netts, list):
#        netts = [netts]
#    for nett in netts:
#        if nett is not None:
#            for param in nett.parameters():
#                param.requires_grad = requires_grad
# NOTE training loop
ganIterations = 0

for epoch in range(opt.niter):
#    if epoch > opt.annealStart:
#        adjust_learning_rate(optimizerD, opt.lrD, epoch, None, opt.annealEvery)
#        adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)
    for i, data in enumerate(dataloaderhaze, 0):         
        #get img
        netG.train()
#        netG_B.train()
        haze_train = data
        free_train = iter(dataloaderfree).next()
        haze_train,free_train = haze_train.float().cuda(), free_train.float().cuda()
        
        fake_free = netG(haze_train)
#        rec_haze = netG_B(fake_free)
#        fake_haze = netG_B(free_train)
#        rec_free = netG_A(fake_haze)
        generate_imgf_A=netG(free_train)        
#        generate_imgf_B=netG_B(haze_train)       
#        fake_free,rec_haze,fake_haze,rec_free = fake_free.cuda(),rec_haze.cuda(),fake_haze.cuda(),rec_free.cuda()
        '''netG'''
        for p in netD_A.parameters():
            p.requires_grad = False
        for p in netD_B.parameters():
            p.requires_grad = False
        optimizerG.zero_grad()    
        '''G_A GAN loss'''
        output_A = netD_A(fake_free)
        label_A = torch.FloatTensor(output_A.size()).fill_(free_label).cuda()
        errGan_A_1 = criterionMSE(output_A, label_A)
        errGan_A = errGan_A_1
        '''G_A G_B perceptual loss'''
        
        features_content_A = vgg(haze_train)
        f_A = Variable(features_content_A[1].data, requires_grad=False)
        features_A = vgg(fake_free)
        errLp_A =  criterionMSE(features_A[1], f_A)

#
        errc_A =  criterionCycle(generate_imgf_A, free_train)   
#        errc_B =  criterionCycle(generate_imgf_B, haze_train)   
#        errG_A = errGan_A + 0.1*errGan_B +err_cycle_A+ 0.1*err_cycle_B+errLp_A+0.1*errLp_B + errc_A+ 0.1*errc_B
#        errG_B = 0.1*errGan_A + errGan_B +0.1*err_cycle_A+ err_cycle_B+0.1*errLp_A+errLp_B + 0.1*errc_A+ errc_B
        errG = errGan_A+errc_A + 0.1*errLp_A
#        errG = errGan_A + errGan_B +err_cycle_A+ err_cycle_B + 0.0003*errLp_A+0.0003*errLp_B+errc_A+ errc_B
#        errG_A.backward(retain_graph=True)
        errG.backward()
#        print('vgg grad')
#        for p in vgg.parameters():
#            print(p.data)
        optimizerG.step()
#        print('vgg grad')
#        for p in vgg.parameters():
#            print(p.data)
        '''netD'''
        for p in netD_A.parameters():
            p.requires_grad = True
        for p in netD_B.parameters():
            p.requires_grad = True
        optimizerD.zero_grad() 
        '''netD_A'''
        output_A_1 = netD_A(free_train)
        label_A_1 = torch.FloatTensor(output_A_1.size()).fill_(free_label).cuda()
        errD_A_real = criterionMSE(output_A_1,label_A_1)      
#        fake_free = netG(haze_train)
        output_A_2 = netD_A(haze_train)
        label_A_2 = torch.FloatTensor(output_A_2.size()).fill_(haze_label).cuda()
        errD_A_fake = criterionMSE(output_A_2,label_A_2)
        
        fake_free = netG(haze_train)
        output_A_3 = netD_A(fake_free)
        label_A_3 = torch.FloatTensor(output_A_3.size()).fill_(haze_label).cuda()
        errD_A_fake_2 = criterionMSE(output_A_3,label_A_3)        
##        generate_imgf_A=netG(free_train)
#        output_A_3 = netD_A(fake_free)
#        output_B_1 = netD_B(fake_free).detach()
##        label3 = torch.FloatTensor(output_3.size()).fill_(haze_label).cuda()
#        errD_AB = criterionMSE(output_A_3,output_B_1)    

#        generate_imgf_B=netG_B(haze_train)
#        output_4 = netD(generate_imgf_B)
#        label3_B = torch.FloatTensor(output_4.size()).fill_(free_label).cuda()
#        errD_F_B = criterionMSE(output_4,label3_B)  
        errD_A = errD_A_real + errD_A_fake+errD_A_fake_2
#        errD_A = errD_A_real + errD_A_fake+errD_AB
        '''netD_B'''
#        output_B_1 = netD_B(haze_train)
#        label_B_1 = torch.FloatTensor(output_B_1.size()).fill_(haze_label).cuda()
#        errD_B_real = criterionMSE(output_B_1,label_B_1)    
#    
#        output_B_3 = netD_B(free_train)
#        label_B_3 = torch.FloatTensor(output_B_3.size()).fill_(free_label).cuda()
#        errD_B_free = criterionMSE(output_B_3,label_B_3)         
#        errD_B = errD_B_real + errD_B_fake       
#        errD_B = errD_B_real + errD_B_free  
#        errD=errD_A+errD_B
        errD=errD_A
        errD_A.backward()
        
#        errD_B.backward()
        
#        print('netD grad')
#        for p in netG_A.parameters():
#            print(p.grad)
#        for p in netG_A.parameters():
#            print('netG parameters before')
#            print(p)
#            break
        optimizerD.step()
#        for p in netG_A.parameters():
#            print('netG parameters after')
#            print(p)
#            break
        ganIterations += 1
    
        if ganIterations % opt.display == 0:
            print('[%d/%d][%d/%d]\
                  L_G_A: %f L_C_A: %f \
                  L_D_A_R: %f L_D_A_F: %f \
                  L_G_A: %f L_D_A: %f '
              % (epoch, opt.niter, i, len(dataloaderhaze),
                 errGan_A.data[0],errc_A.data[0],              
                 errD_A_real.data[0], errD_A_fake.data[0],
                 errG.data[0],errD_A.data[0]))
            sys.stdout.flush()
            trainLogger.write('%d\t%f\t%f\t%f\t%f\t%f\n' % \
                            (i,errGan_A.data[0],                
                 errD_A_real.data[0], errD_A_fake.data[0],
                 errG.data[0],errD_A.data[0]))
            trainLogger.flush()
    
        if epoch % 2 == 0:            
            torch.save(netG.state_dict(), '%s/netG_singleHFepoch_%d.pth' % (opt.exp, epoch))
#            torch.save(netD.state_dict(), '%s/netDepoch_%d.pth' % (opt.exp, epoch))
        if ganIterations % opt.evalIter == 0:  
            val_batch_output = torch.FloatTensor(val_input.size()).fill_(0)
            for idx in range(val_input.size(0)):
                single_img = val_input[idx,:,:,:].unsqueeze(0)
                val_inputv = Variable(single_img, volatile=True)
                netG.eval()
                dehaze21 = netG(val_inputv)
                val_batch_output[idx,:,:,:].copy_(dehaze21[0,:,:,:])
                del dehaze21
                del single_img
                del val_inputv
            vutils.save_image(val_batch_output, '%s/generated_epoch_%08d_iter%08d.png' % \
                    (opt.exp, epoch, ganIterations), normalize=True, scale_each=False)
            del val_batch_output

          
trainLogger.close()
