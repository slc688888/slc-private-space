import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import glob
import scipy.ndimage
import colorsys 
import h5py
import torchvision.transforms as transforms
import re
import random
import torch
import torch.nn.functional as F
from yolov3.utils.augmentations import horisontal_flip
#from skimage import transform
#import cv2
IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image
def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
  images = []
  if not os.path.isdir(dir):
    raise Exception('Check dataroot')
  for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
      if is_image_file(fname):
        path = os.path.join(dir, fname)
        item = path
        images.append(item)
  return images

def default_loader(path):
  return Image.open(path).convert('RGB')
def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad
def HSVColor(img):
    r,g,b = img.split()
    Hdat = []
    Sdat = []
    Vdat = [] 
    for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
        h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
        Hdat.append(int(h))
        Sdat.append(int(s))
        Vdat.append(int(v))
    r.putdata(Hdat)
    g.putdata(Sdat)
    b.putdata(Vdat)
    return Image.merge('RGB',(r,g,b))
class Read_Dhaze(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, seed=None):
    self.root = root
    self.transform = transform
    self.loader = loader
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):
    f = open("RTTS_train.txt","r")
    train_list=f.read()
    f.close()
    train_list=train_list.split('\n') 
    name='./'+train_list[index]
    GT_name=name.replace('JPEGImages','dehaze')
    label_name='./'+train_list[index]
    label_name=label_name.replace('png','txt')
    boxes = torch.from_numpy(np.loadtxt(label_name).reshape(-1, 5))
    img=Image.open(name)
#    img_GT=Image.open(GT_name)
#    r2,g2,b2=img_GT.split()
    r,g,b= img.split()
    bgr=transforms.ToTensor()(img)
    img=transforms.ToTensor()(img)
#    bgr_GT=transforms.ToTensor()(img_GT)
#    bgr=transforms.ToTensor()(Image.merge('RGB',(b,g,r)))
#    bgr_GT=transforms.ToTensor()(Image.merge('RGB',(b2,g2,r2)))
#    print(bgr[0,256,256])
    _, h, w = bgr.shape
#    print(h,w,'\n')
    h_factor, w_factor = (h, w)
    # Pad to square resolution
    bgr, pad = pad_to_square(bgr, 0)
#        print(pad.shape)
    _, padded_h, padded_w = bgr.shape

    # ---------
    #  Label
    # ---------

#    label_path = self.label_files[index % len(self.img_files)].rstrip()

    targets = None
    if os.path.exists(label_name):
        boxes = torch.from_numpy(np.loadtxt(label_name).reshape(-1, 5))
        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
#        print(x1,x2,y1,y2,'\n')
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
#        print(x1,x2,y1,y2,'\n')
#        print(padded_w,padded_h,'\n')
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w
        boxes[:, 4] *= h_factor / padded_h

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes
#    if np.random.random() < 0.5:
#        bgr, targets = horisontal_flip(bgr, targets)
#        if self.transform is not None:
#            bgr= self.transform(bgr)
    bgr = resize(bgr, 512)
    return img,bgr,targets,name#,bgr_GT
#    h=bgr.size[0]
#    w=bgr.size[1]
#    new_w = int(w * min(512/w, 512/h))
#    new_h = int(h * min(512/w, 512/h))    
#    bgr = bgr.resize((new_h, new_w),Image.ANTIALIAS)
#    
#    canvas = np.full((512,512,3), 0)
#
#    canvas[(512-new_w)//2:(512-new_w)//2 + new_w,(512-new_h)//2:(512-new_h)//2 + new_h,  :] = bgr
#    canvas=canvas/255
#    if self.transform is not None:
#      img= self.transform(canvas)
#
#    return img,name,labels[:-1]

  def __len__(self):
    f = open("RTTS_train.txt","r")
    train_list=f.read()
    train_list=train_list.split('\n') 
    f.close()
    return len(train_list)-2