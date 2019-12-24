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
from torch.autograd import Variable
from skimage.measure import compare_ssim as ssim
import misc
import models.DenseUnet as net
from myutils.vgg16 import Vgg16
from myutils import utils
import glob
from PIL import Image
import torch.optim.lr_scheduler as lr_scheduler
import skimage
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='datatest',  help='')
parser.add_argument('--dataroot', required=False,
  default='./ITS/hazy', help='path to trn dataset')
parser.add_argument('--datarootfree', required=False,
  default='./D_HAZE/NYU_GT/', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='./SOTS/hazy', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=416, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=416, help='the height / width of the cropped input image to network')
parser.add_argument('--condition_GAN', type=int,
  default=1, help='set to 0 to use unconditional discriminator')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=48)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=40, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lamdaA', type=float, default=0.0066, help='lambdaGAN')
parser.add_argument('--lamdaP', type=float, default=1, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='DADN_ITS_new', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=50, help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()
print(opt)

misc.create_exp_dir(opt.exp)
opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
opt.workers=1

'''get dataloader'''
dataloaderhaze = misc.getLoader(opt.dataset,
                       opt.dataroot,
                       opt.originalSize,
                       opt.imageSize,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='train',
                       shuffle=False,
                       seed=opt.manualSeed)
dataloadertest = misc.getLoader(opt.dataset,
                       opt.valDataroot,
                       opt.originalSize,
                       opt.imageSize,
                       opt.valBatchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='endtoend',
                       shuffle=False,
                       seed=opt.manualSeed)
trainLogger = open('%s/train.log' % opt.exp, 'w')

'''para'''
real_label = 1
fake_label = 0
ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize
idx_A = {inputChannelSize+1, inputChannelSize+outputChannelSize}
idx_B = {1, inputChannelSize}

'''init netG,netD'''
netG = net.G_HDC()
misc.init_weights(netG)
print(netG)
netD_A = net.D(inputChannelSize,outputChannelSize, ndf,3)
netD_A.apply(misc.weights_init)
if opt.netD != '':
  netD_A.load_state_dict(torch.load(opt.netD_A))
print(netD_A)

'''loss function'''
criterionBCE = nn.BCELoss()
criterionMSE = nn.MSELoss()
criterionL1 = nn.L1Loss()
errD, errG, errL1 = 0, 0, 0
netD_A.cuda()
netG.cuda()
criterionBCE.cuda()
criterionMSE.cuda()
criterionL1.cuda()
lamdaA = opt.lamdaA
lamdaP = opt.lamdaP
print('init done')


'''Initialize VGG-16'''
vgg = Vgg16()
vgg.load_state_dict(torch.load(os.path.join('./models/', "vgg16.weight")))
vgg.eval()
vgg.cuda()

''' set optimizer'''
optimizerD = optim.Adam(netD_A.parameters(), lr = opt.lrD, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999))
schedulerG = lr_scheduler.MultiStepLR(optimizerG, milestones=[round(opt.niter * x) for x in [0.8, 0.9]], gamma=0.5)
schedulerD = lr_scheduler.MultiStepLR(optimizerD, milestones=[round(opt.niter * x) for x in [0.8, 0.9]], gamma=0.5)
ganIterations = 0
for f in glob.glob('results.txt'):
    os.remove(f)
for epoch in range(opt.niter):
    i=0
    psnr_count=0
    ssim_count=0
    for data_hazy,data_free in dataloaderhaze:         
        #get img
        i+=1
        netG.train()
        haze_train1 = data_hazy
        free_train1 = data_free
        psnr_count_free=0
        ssim_count_free=0
        haze_train1,free_train1 = haze_train1.float().cuda(), free_train1.float().cuda()       
#        haze_train = haze_train1.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#        im=Image.fromarray(haze_train)
#        im.show()
#        free_train = free_train1.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#        im=Image.fromarray(free_train)
#        im.show()

        
        fake_free = netG(haze_train1)
        generate_imgf_A=netG(free_train1)#.detach()
        '''netG'''
        for p in netD_A.parameters():
            p.requires_grad = False
        optimizerG.zero_grad()    
        '''G GAN loss'''
        output_A = netD_A(fake_free)        
        label_A = torch.FloatTensor(output_A.size()).fill_(real_label).cuda()
        errGan_A_1 = criterionBCE(output_A, label_A)
        errGan_A = errGan_A_1
        '''G perceptual loss'''        
        features_content_A = vgg(free_train1)
        f_A_1 = Variable(features_content_A[0].data, requires_grad=False)
        f_A_2 = Variable(features_content_A[1].data, requires_grad=False)
        features_A = vgg(fake_free)
        errLp_A_1 =  criterionMSE(features_A[0], f_A_1)
        errLp_A_2 =  criterionMSE(features_A[1], f_A_2)
#        f_A = f_A.squeeze(0).permute(1, 2, 0).to('cpu', torch.uint8).numpy()[:,:,-1]
#    
#        plt.imshow(f_A, cmap=plt.cm.gray)
#        plt.savefig("gray.png")        
#        plt.show() 
#        
        errLp_A =  0.5*errLp_A_1 + 0.1*errLp_A_2
        '''G free-free pixel loss'''
        errc_A_F = criterionL1(generate_imgf_A, free_train1)
        '''G hazy-free pixel loss'''
        errc_A_P =  criterionL1(fake_free, free_train1)
        '''G edge loss '''
        erre = criterionL1(misc.edge_compute(fake_free),misc.edge_compute(free_train1))
        '''G loss'''
        errG = errc_A_P + errLp_A+0.1*errGan_A+erre +errc_A_F
        errG.backward()
        optimizerG.step()
        
      
        '''netD'''
        for p in netD_A.parameters():
            p.requires_grad = True
        optimizerD.zero_grad() 
        '''netD real'''
        output_A_1 = netD_A(free_train1)
        label_A_1 = torch.FloatTensor(output_A_1.size()).fill_(real_label).cuda()
        errD_A_real = criterionBCE(output_A_1,label_A_1)      
        '''netD fake'''
        fake_free = netG(haze_train1).detach()
        output_A_2 = netD_A(fake_free)        
        label_A_2 = torch.FloatTensor(output_A_2.size()).fill_(fake_label).cuda()
        errD_A_fake = criterionBCE(output_A_2,label_A_2)
        '''D loss'''
        errD_A = errD_A_real + errD_A_fake
        errD=errD_A
        errD_A.backward()
        optimizerD.step()

        '''output log'''
        ganIterations += 1  
        if ganIterations % opt.display == 0:
            print('[%d/%d][%d/%d]\
                  errGan_A: %f errc_A_F: %f errc_A_P: %f errLp_A: %f erre: %f\
                  errD_A_real: %f errD_A_fake: %f\
                  errG: %f errD_A: %f'
              % (epoch, opt.niter, i, len(dataloaderhaze),
                 errGan_A.data,errc_A_F.data,errc_A_P.data, errLp_A.data, erre.data,             
                 errD_A_real.data, errD_A_fake.data,
                 errG.data,errD_A.data))
            sys.stdout.flush()
            trainLogger.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % \
                            (i,               
                 errGan_A.data,errc_A_F.data,errc_A_P.data, errLp_A.data,erre.data,          
                 errD_A_real.data, errD_A_fake.data,
                 errG.data,errD_A.data))
            trainLogger.flush()
           
    torch.save(netG.state_dict(), '%s/netG_%d.pth' % (opt.exp, epoch))
    netG.eval()
    for test_hazy,test_free in dataloadertest: 
        test_hazy=test_hazy.float().cuda()
        test_free2=test_free.float().cuda()
        with torch.no_grad():
            dehaze_test = netG(test_hazy)
            dehaze_free = netG(test_free2)
        dehaze_test = dehaze_test.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        test_free = test_free.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()    
        dehaze_free = dehaze_free.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        psnr_count+=skimage.measure.compare_psnr(dehaze_test,test_free,255)
        ssim_count+=ssim(dehaze_test,test_free,data_range=255,multichannel=True)
        psnr_count_free+=skimage.measure.compare_psnr(dehaze_free,test_free,255)
        ssim_count_free+=ssim(dehaze_free,test_free,data_range=255,multichannel=True)     
    with open('results.txt', 'a') as f:
        f.write('epoch'+str(epoch)+'\t'+'psnr='+str(psnr_count/len(dataloadertest))+'\t'+'ssim='+str(ssim_count/len(dataloadertest)) + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)          
    schedulerD.step()
    schedulerG.step()
trainLogger.close()
