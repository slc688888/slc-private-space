#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 01:05:26 2019

@author: littlemonster
"""

import os

#查找文件
path="facades/train256_new2"
#os.listdir()方法，列出来所有文件
#返回path指定的文件夹包含的文件或文件夹的名字的列表
files=os.listdir(path)

#主逻辑
#对于批量的操作，使用FOR循环
for f in files:
    #调试代码的方法：关键地方打上print语句，判断这一步是不是执行成功
    print(f)
    name = int(f[:-3])
    if name <= 1000:        
        rename = str(name+1000)+f[-3:]
        new_file=os.path.join(path,rename)
        old_file=os.path.join(path,f)
        os.rename(old_file,new_file)
#    if "project20" in f and f.endswith(".jpg"):
#        print("原来的文件名字是:{}".format(f))
#
#        #找到老的文件所在的位置
#        old_file=os.path.join(path,f)
#        print("old_file is {}".format(old_file))
#        #指定新文件的位置，如果没有使用这个方法，则新文件名生成在本项目的目录中
#        new_file=os.path.join(path,"project30.jpg")
#        print("File will be renamed as:{}".format(new_file))
#        os.rename(old_file,new_file)
#        print("修改后的文件名是:{}".format(f))
#
#
#
#
#import os,sys
#
#path="C:\\Users\\Jw\\Desktop\\python_work"
## 切换到 对应 目录
#os.chdir(path )
#
##列出目录
#print("目录为：%s"%os.listdir(os.getcwd()))
#
##移除
#os.remove("project30.jpg")
#
##移除后的目录
#print("移除后：%s"%os.listdir(os.getcwd()))