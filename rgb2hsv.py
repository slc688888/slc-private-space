#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:09:53 2018

@author: littlemonster
"""

from PIL import Image
import colorsys    
import os, shutil
import h5py
import numpy
def HSVColor(img):
    if isinstance(img,Image.Image):
        r,g,b = img.split()
        Hdat = []
        Sdat = []
        Vdat = [] 
        for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
#            h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
            r1,g1,b1 = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
            Hdat.append(r1*255.)
            Sdat.append(g1*255.)
            Vdat.append(b1*255.)
#            Hdat.append(int(h*255.))
#            Sdat.append(int(s*255.))
#            Vdat.append(int(v*255.))
#        r.putdata(Hdat)
#        g.putdata(Sdat)
#        b.putdata(Vdat)
#        return Image.merge('RGB',(r,g,b))
        return Hdat,Sdat,Vdat
    else:
        return None
    
#a = Image.open('a.jpg')
#h,s,v = numpy.array(HSVColor(a))
##h = h.reshape(numpy.array(a).shape[0],numpy.array(a).shape[1])
##s = s.reshape(numpy.array(a).shape[0],numpy.array(a).shape[1])
##v = v.reshape(numpy.array(a).shape[0],numpy.array(a).shape[1])
#f=h5py.File("myh5py.hdf5","w")
#d1=f.create_dataset("h",h.shape, 'f')
#d2=f.create_dataset("s",s.shape, 'f')
#d3=f.create_dataset("v",v.shape, 'f')
#d4=f.create_dataset('high',data=numpy.array(a).shape[0])
#d5 = f.create_dataset('wide',data=numpy.array(a).shape[1])
#d1[...] = h
#d2[...] = s
#d3[...] = v
#for key in f.keys():
#    print(f[key].name)
#    print(f[key].value)
def batchRenameFile(srcDirName, destDirName):  # srcDirName 为源文件夹的绝对路径，真正保存数据文件的子文件夹都在该文件夹下；destDirName 为目标文件夹的绝对路径
    subDirNameList = os.listdir(srcDirName)  # 获取真正保存数据文件的文件夹序列
    for subDirName in subDirNameList:
        fileList = os.listdir(srcDirName+'/'+subDirName)    # 此处须给出绝对路径
        deslist = destDirName+'/'+subDirName
        os.makedirs(deslist)
        i = 0;
        count=0
        for file in fileList:
            if file.endswith('.jpg') or file.endswith('.JPG'):
                deslist = destDirName+'/'+subDirName+'/'+str(i-count)+'.hdf5'
                a = Image.open(srcDirName+'/'+subDirName+'/'+file)
                b= numpy.array(a).shape[-1]
                if b ==3:                    
                    h,s,v = numpy.array(HSVColor(a))
                    f=h5py.File(deslist,"w")
                    d1=f.create_dataset("h",h.shape, 'f')
                    d2=f.create_dataset("s",s.shape, 'f')
                    d3=f.create_dataset("v",v.shape, 'f')
                    d4=f.create_dataset('high',data=numpy.array(a).shape[0])
                    d5 = f.create_dataset('wide',data=numpy.array(a).shape[1])
                    d1[...] = h
                    d2[...] = s
                    d3[...] = v
                    f.close()
                    del d1
                    del d2
                    del d3
                    del a
                    del h
                    del s
                    del v
                else:
                    count+=1
#                shutil.copy(srcDirName+'/'+subDirName+'/'+file, deslist)  # 此处须给出绝对路径
                i = i+1
srcDirName = '/home/littlemonster/Desktop/slc-private-space-master/light-haze'
destDirName = '/home/littlemonster/Desktop/slc-private-space-master/image_HSV'
batchRenameFile(srcDirName, destDirName)
#a = Image.open('60.jpg')

