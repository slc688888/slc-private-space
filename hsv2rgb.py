#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:09:53 2018

@author: littlemonster
"""

from PIL import Image
import colorsys    
import h5py
import numpy
def RGBColor(h,s,v,high,wide):
    R = []
    G = []
    B = [] 
    for rd,gn,bl in zip(h,s,v) :
        r1,g1,b1 = colorsys.hsv_to_rgb(rd/255.,gn/255.,bl/255.)
        R.append(int(r1*255.))
        G.append(int(g1*255.))
        B.append(int(b1*255.))
    r = Image.fromarray(numpy.array(R).reshape(high,wide).astype(numpy.uint8)).convert('L')
    g = Image.fromarray(numpy.array(G).reshape(high,wide).astype(numpy.uint8)).convert('L')
    b = Image.fromarray(numpy.array(B).reshape(high,wide).astype(numpy.uint8)).convert('L')
    rgb = Image.merge('RGB',(r,g,b))
    return rgb


#HDF5的读取：.jpg')
f = h5py.File('2.hdf5','r')   #打开h5文件
h=f['h'].value
s=f['s'].value
v = f['v'].value
high = f['high'].value
wide = f['wide'].value
hsv = numpy.concatenate((h, s, v)).reshape(3,high,wide)
hsv = hsv.astype(numpy.uint8)

r = Image.fromarray(hsv[0]).convert('L')
g = Image.fromarray(hsv[1]).convert('L')
b = Image.fromarray(hsv[2]).convert('L')
image = Image.merge("RGB", (r, g, b))
#image.show()
image.save('sample/test.jpg')
#b= RGBColor(h,s,v,high,wide)
#b.save('b.jpg')