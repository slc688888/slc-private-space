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
from skimage.measure import compare_ssim as ssim
import skimage
def psnr(im1,im2):
    rmse = np.mean((im1 - im2)**2)
    psnr = 20*math.log10(255/(rmse**0.5))
    return psnr

gtroot='./SOTS/clear/'
dhroot='./SOTS/hazy/'
dehazeroot='./compare_new/DADN_sots/'
list_free=glob.glob('./SOTS/clear/*.png')
list_hazy=glob.glob('./compare_new/DADN_sots/*.png')
list_free.sort()
haze_psnr=0
dehaze_psnr=0
haze_ssim=0
dehaze_ssim=0
count=0
#for i in list_free:     
#    free=np.array(Image.open(i))
#    for ii in range(1,11):
#        dehaze=np.array(Image.open(dehazeroot+str(1400+int(count/10))+'_'+str(ii)+'.png'))
#        dehaze_psnr+=skimage.measure.compare_psnr(dehaze, free, 255)
#        dehaze_ssim+=ssim(dehaze,free,data_range=255,multichannel=True)
#    count+=10
#    print(i)
#    count+=1
for i in list_free:     
    free=np.array(Image.open(i))
    dehaze=np.array(Image.open(dehazeroot+str(1400+int(count/10))+'_'+str(10)+'.png'))
    dehaze_psnr+=skimage.measure.compare_psnr(dehaze,free,255)
    dehaze_ssim+=ssim(dehaze,free,data_range=255,multichannel=True)
    count+=10
    print(i)
haze_psnr=haze_psnr/len(list_hazy)
dehaze_psnr=dehaze_psnr/len(list_hazy)
haze_ssim=haze_ssim/len(list_hazy)
dehaze_ssim=dehaze_ssim/len(list_hazy)

