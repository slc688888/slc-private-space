#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:58:21 2019

@author: littlemonster
"""
from PIL import Image
import glob
import numpy as np
import math
import skimage
from skimage.measure import compare_ssim as ssim
def psnr(im1,im2):
    im1,im2=im1.astype(np.float64),im2.astype(np.float64)
    rmse = np.mean(np.square(im1 - im2),dtype=np.float64)
    psnr = 10*np.log10((255**2)/(rmse))
    return psnr

gtroot='./compare_new/gt/'
dhroot='./compare_new/DADN_256_baseline_vgg_gan/'
list_free=glob.glob('./compare_new/DADN_256_baseline_vgg_gan/*.png')

haze_psnr=0
dehaze_psnr=0
haze_ssim=0
dehaze_ssim=0
for i in range(len(list_free)):     
    free=np.array(Image.open(gtroot+str(i)+'.png'))
    dehaze=np.array(Image.open(dhroot+str(i)+'.png'))
    dehaze_psnr+=skimage.measure.compare_psnr(dehaze,free)
#    dehaze_psnr+=psnr(dehaze,free)
    dehaze_ssim+=ssim(dehaze,free,data_range=255,multichannel=True)
    print(i)
haze_psnr=haze_psnr/len(list_free)
dehaze_psnr=dehaze_psnr/len(list_free)
haze_ssim=haze_ssim/len(list_free)
dehaze_ssim=dehaze_ssim/len(list_free)

#gtroot='./test/test11/gt/'
#dhroot='./test/test11/DCP/'
##list_free=glob.glob('./test/test1(copy)/AOD/*.png')
#
#haze_psnr=0
#dehaze_psnr=[]
#haze_ssim=0
#dehaze_ssim=[]
#for i in range(2):     
#    free=np.array(Image.open(gtroot+str(i)+'.png'))
#    dehaze=np.array(Image.open(dhroot+str(i)+'.png'))
#    dehaze_psnr.append(psnr(dehaze,free))
#    dehaze_ssim.append(ssim(dehaze,free,data_range=255,multichannel=True))
#    print(i)
#haze_psnr=haze_psnr/len(list_free)
#dehaze_psnr=dehaze_psnr/len(list_free)
#haze_ssim=haze_ssim/len(list_free)
#dehaze_ssim=dehaze_ssim/len(list_free)    