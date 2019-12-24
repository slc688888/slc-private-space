#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:53:17 2018
@author: littlemonster
"""

"""
******important******

PIL read rgb,opencv read bgr

"""


a=1
b=1
c=a+b
from PIL import Image
import numpy
import torchvision.transforms as transforms
import cv2
img=Image.open('./D_HAZE/NYU_Hazy/'+'2'+'_Hazy.bmp')
im = cv2.imread('./D_HAZE/NYU_Hazy/'+'2'+'_Hazy.bmp')
##img.show() 
##img.save('test.jpg')
#img2=Image.open('test.jpg')
#img2_bands=img2.getbands()
#img_bands = img.getbands()
r,g,b= img.split()
bgr=Image.merge('RGB',(b,g,r))
img_r = Image.merge('RGB',(r,g.point(lambda i:i==i*0),b.point(lambda i:i==i*0)))#red
img_r.show()
img_g = Image.merge('RGB',(r.point(lambda i:i==i*0),g,b.point(lambda i:i==i*0)))#green
img_g.show()
img_b = Image.merge('RGB',(r.point(lambda i:i==i*0),g.point(lambda i:i==i*0),b))#blue
img_b.show()
#img2=transforms.ToTensor()(img)
#img2.permute((2,1,0))
#img3=transforms.ToPILImage()(img2)
