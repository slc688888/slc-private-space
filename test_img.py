#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 02:50:38 2018

@author: littlemonster
"""

from PIL import Image
import numpy
count=0
s = []
for i in range(517):
    img=Image.open('/home/littlemonster/Desktop/slc-private-space-master/image/dense-haze/'+str(i)+'.jpg')
    img1 = numpy.array(img)
    a = img1.shape
    if a[-1] != 3:
        count+=1
        s.append(i)
    