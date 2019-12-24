#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:50:51 2019

@author: littlemonster
"""
import cv2 as cv
import glob
# 读入原图片
resize_list=glob.glob('O-HAZY/GT/*')
for path in resize_list:    
    img = cv.imread(path)
    # 打印出图片尺寸
    print(img.shape)
    # 将图片高和宽分别赋值给x，y
    x, y = img.shape[0:2]
      
    # 显示原图
    #cv.imshow('OriginalPicture', img)
      
    # 缩放到原来的二分之一，输出尺寸格式为（宽，高）
#    img_test1 = cv.resize(img, (int(y / 2), int(x / 2)))
    #cv.imshow('resize0', img_test1)
    #cv.waitKey()
      
    # 最近邻插值法缩放
    # 缩放到原来的四分之一
    img_test2 = cv.resize(img, (0, 0), fx=0.1, fy=0.1)
    cv.imwrite(path.replace('GT','GT_resize'), img_test2)


