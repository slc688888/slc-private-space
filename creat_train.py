#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:33:32 2019

@author: littlemonster
"""

from __future__ import division
import numpy as np
import sys

sys.path.append("./mingqingscript")

import scipy.io as sio
import scipy.ndimage.interpolation
# import scipy.signal

import os

import math
import random

import pdb
import random
import numpy as np
import pickle
import random
import sys
import shutil

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

# torch condiguration
import argparse
from math import log10
# import scipy.io as sio
import numpy as np

import random
from random import uniform
import h5py
import time
import PIL
from PIL import Image

import h5py
import numpy as np
import matplotlib.pyplot as plt

# plt.axis([0, 10, 0, 1])
plt.ion()


# for i in range(10):
#     y = np.random.random()
#     plt.scatter(i, y)
#     plt.pause(0.05)
#
# while True:
#     plt.pause(0.05)
def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255 * np.ones((len(arr), 1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


index = 1
nyu_depth = h5py.File('nyu_depth_v2_labeled.mat', 'r')

directory='facades/train416_more'

if not os.path.exists(directory):
    os.makedirs(directory)
directory='facades/test416_more'

if not os.path.exists(directory):
    os.makedirs(directory)

image = nyu_depth['images']
depth = nyu_depth['depths']

img_size = 256

# per=np.random.permutation(1400)
# np.save('rand_per.py',per)
# pdb.set_trace()
total_num = 0
plt.ion()
for index in range(image.shape[0]):
#    index = index
#    gt_image = (image[index, :, :, :]).astype(float)
    gt_image = (image[index, :, :, :]).astype(float)    
    gt_image = np.swapaxes(gt_image, 0, 2)
#    w=gt_image.shape[0]-6
#    h=gt_image.shape[1]-6
    gt_image_withoutwite=gt_image[8:gt_image.shape[0]-8,8:gt_image.shape[1]-8,:]
#    a = np.zeros(shape=(3,w,h))
#    im = Image.open("b.jpg")
#    
#    img = np.array(im)
#    im=Image.fromarray(gt_image)
#    im.show() 
    gt_image = scipy.misc.imresize(gt_image_withoutwite, [img_size, img_size]).astype(float)

    gt_image = gt_image / 255


    gt_depth = depth[index, :, :]
    maxhazy = gt_depth.max()
    minhazy = gt_depth.min()
    gt_depth = (gt_depth) / (maxhazy)

    gt_depth = np.swapaxes(gt_depth, 0, 1)
    scale1 = (gt_depth.shape[0]) / img_size
    scale2 = (gt_depth.shape[1]) / img_size

    gt_depth = scipy.ndimage.zoom(gt_depth, (1 / scale1, 1 / scale2), order=1)

    if gt_depth.shape != (img_size, img_size):
        continue
    
    beta=np.ones((gt_depth.shape[0],gt_depth.shape[1]))
    for i in range(beta.shape[0]):
#        m=uniform(0.4, 1.6)
        for j in range(beta.shape[1]):
            beta[i][j] = uniform(0.4, 1.6)
            
#    beta = uniform(0.4, 1.6)
#    beta = uniform(0.4, 1.6)
    tx1 = np.exp(-beta * gt_depth)
    tx1=np.reshape(tx1,(tx1.shape[0],tx1.shape[1],1))
    
    A=np.ones((gt_image.shape[0],gt_image.shape[1]))
    for i in range(A.shape[0]):
#        m=uniform(0.5, 1)
        for j in range(A.shape[1]):
            A[i][j] = uniform(0.5, 1)
    A=np.reshape(A,(A.shape[0],A.shape[1],1))

    a = 1 - 0.5 * uniform(0, 1)
    A=np.zeros(shape=(1, 1, 3))
    A=A+a

    m = gt_image.shape[0]
    n = gt_image.shape[1]

    rep_atmosphere = np.tile(A, [m, n, 1])
    tx1 = np.reshape(tx1, [m, n, 1])
    max_transmission = np.tile(tx1, [1, 1, 3])
#    
    
#    rep_atmosphere = np.tile(A, [1, 1, 3])
#    max_transmission = np.tile(tx1, [1, 1, 3])

    haze_image = gt_image * max_transmission + rep_atmosphere * (1 - max_transmission)
    
#        haze_image=np.array((haze_image*255).astype(np.uint8))
#        im=Image.fromarray(haze_image)
#        im.show() 

    total_num = total_num + 1
    scipy.misc.imsave('a0.9beta1.29.jpg', haze_image)
    scipy.misc.imsave('gt.jpg', gt_image)
    if index < 1000:            
        h5f=h5py.File('./facades/train416_more/'+str(total_num)+'.h5','w')
        h5f.create_dataset('haze',data=haze_image)
        h5f.create_dataset('trans',data=max_transmission)
        h5f.create_dataset('atom',data=rep_atmosphere)
        h5f.create_dataset('gt',data=gt_image)
    else: 
        h5f=h5py.File('./facades/test416_more/'+str(total_num - 999)+'.h5','w')
        h5f.create_dataset('haze',data=haze_image)
        h5f.create_dataset('trans',data=max_transmission)
        h5f.create_dataset('atom',data=rep_atmosphere)
        h5f.create_dataset('gt',data=gt_image)