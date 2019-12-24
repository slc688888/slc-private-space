import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import glob
import scipy.ndimage
import colorsys 
import h5py
import cv2
import torch
IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

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
  def __init__(self, root,data, transform=None, loader=default_loader, seed=None):
    self.root = root
    self.transform = transform
    self.loader = loader
    self.data=data

    if seed is not None:
      np.random.seed(seed)

#  def __getitem__(self, index):
##    name=''
##    if self.data == 'haze':
###        name='.jpg'
##        name='_Hazy.bmp'
##        index=index+704
##    else:
##        name='_Image_.bmp'
###        name='.jpg'
#    train_list=glob.glob(self.root+'/*.bmp')
#    train_list.sort()
#    img=Image.open(train_list[index])
##    print(train_list[index])
##    img=Image.open(self.root+str(index+1)+name)#RGB
##    img=Image.open(self.root+str(index)+name)#RGB
#    r,g,b= img.split()
#    bgr=Image.merge('RGB',(b,g,r))
#
#    if self.transform is not None:
#      img= self.transform(bgr)
#      
#    img2 = Image.open('./D_HAZE/NYU_GT/'+train_list[index][17:-9]+'_Image_.bmp')
##    print('./D_HAZE/NYU_GT/'+train_list[index][17:-9]+'_Image_.bmp')
#    r,g,b= img2.split()
#    bgr=Image.merge('RGB',(b,g,r))
#
#    if self.transform is not None:
#      img2= self.transform(bgr)    
#    return img,img2
  def __getitem__(self, index):

    file_name=self.root+'/'+str(index+1)+'.h5'
    f=h5py.File(file_name,'r')

    haze_image=f['haze'][:]
    trans_map=f['trans'][:]
    ato_map=f['atom'][:]
    GT=f['gt'][:]


#    haze_image= (haze_image - 0.5)*2
#    GT= (GT - 0.5)*2
    haze_image=np.swapaxes(haze_image,0,2)
    trans_map=np.swapaxes(trans_map,0,2)
    ato_map=np.swapaxes(ato_map,0,2)
    GT=np.swapaxes(GT,0,2)



    haze_image=np.swapaxes(haze_image,1,2)
    trans_map=np.swapaxes(trans_map,1,2)
    ato_map=np.swapaxes(ato_map,1,2)
    GT=np.swapaxes(GT,1,2) 
#    trans_map[0] = haze_image[2]
#    trans_map[1] = haze_image[1]
#    trans_map[2] = haze_image[0]
#    
#    ato_map[0] = GT[2]
#    ato_map[1] = GT[1]
#    ato_map[2] = GT[0]
    
#    return trans_map,ato_map
    return haze_image,GT
  def __len__(self):
#    train_list=glob.glob(self.root+'/*.bmp')
    train_list=glob.glob(self.root+'/*.h5')
    return len(train_list)
