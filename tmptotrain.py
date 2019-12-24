#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:09:53 2018

@author: littlemonster
"""


import os, shutil

def batchRenameFile(srcDirName, destDirName):  # srcDirName 为源文件夹的绝对路径，真正保存数据文件的子文件夹都在该文件夹下；destDirName 为目标文件夹的绝对路径
    subDirNameList = os.listdir(srcDirName)
    subDirNameList.sort()# 获取真正保存数据文件的文件夹序列
    for i,subDirName in enumerate(subDirNameList):
#        fileList = os.listdir(srcDirName+'/'+subDirName)    # 此处须给出绝对路径
#        deslist = destDirName+'/'+subDirName
#        os.makedirs(destDirName)
#        i = 0;
#        for file in fileList:
#        if file.endswith('_Image_.bmp') or file.endswith('_Image_.BMP'):
        if subDirName.endswith('.jpg') or subDirName.endswith('.JPG'):
#            deslist = destDirName+'/'+subDirName+'/'+str(i)+'.bmp'
            deslist = destDirName+'/'+str(i)+'.jpg'
#            shutil.copy(srcDirName+'/'+subDirName+'/'+file, deslist)  # 此处须给出绝对路径
            shutil.copy(srcDirName+'/'+subDirName, deslist)
#            i = i+1
            
def batchRenameFile2(srcDirName, destDirName):  # srcDirName 为源文件夹的绝对路径，真正保存数据文件的子文件夹都在该文件夹下；destDirName 为目标文件夹的绝对路径
#    subDirNameList = os.listdir(dirr)
    index=0
    for dirr in srcDirName:
        subDirNameList = os.listdir(dirr)
        for i,subDirName in enumerate(subDirNameList):
            if subDirName.endswith('.jpg') or subDirName.endswith('.JPG'):
                deslist = destDirName+'/'+str(index)+'.jpg'
                shutil.copy(dirr+'/'+subDirName, deslist)
                index+=1

            
#src1DirName = '/home/littlemonster/Desktop/slc-private-space-master/image/dense-haze'
#src2DirName = '/home/littlemonster/Desktop/slc-private-space-master/image/middle-haze'
#src3DirName = '/home/littlemonster/Desktop/slc-private-space-master/image/light-haze'
#srcDirName=[src1DirName,src2DirName,src3DirName]
srcDirName='/home/littlemonster/Desktop/slc-private-space-master/SOTS/outdoor/hazy'
destDirName = '/home/littlemonster/Desktop/slc-private-space-master/SOTS/outdoor/hazy_1'
os.makedirs(destDirName)
batchRenameFile(srcDirName, destDirName)

