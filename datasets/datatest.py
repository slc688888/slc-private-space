import torch
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
def cut_image(image):
    width, height = image.size
    box_list = []

    for i in range(int(height/256)):
        for j in range(int(width/256)):
            box = (j * 256, i * 256, (j + 1) * 256, (i + 1) * 256)
            box_list.append(box)

    image_list = [image.crop(box) for box in box_list]
    return image_list
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

    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):
    train_list=glob.glob(self.root+'/*')
    train_list.sort()
#    print(train_list[index][11:train_list[index].index('_')])
    image=Image.open(train_list[index])#RGB
#    free=Image.open(train_list[index][0:train_list[index].index('_')].replace('hazy','clear')+'.png')
    image=self.transform(image)
#    print(image.size())
#    free=self.transform(free)
#    print(train_list[index])
#    ''''''
#    if np.random.uniform() > 0.5:
#        image=torch.flip(image,[-1])
#        free=torch.flip(free,[-1])
    
#    image = np.array(image)
#    r = image[:,:,0]
#    g = image[:,:,1]
#    b = image[:,:,2]
#    r = np.pad(r, ((0,(int(r.shape[0]/256)+1)*256-r.shape[0]),(0,(int(r.shape[1]/256)+1)*256-r.shape[1])),  'constant', constant_values=(0,0))
#    g = np.pad(g, ((0,(int(g.shape[0]/256)+1)*256-g.shape[0]),(0,(int(g.shape[1]/256)+1)*256-g.shape[1])),  'constant', constant_values=(0,0))
#    b = np.pad(b, ((0,(int(b.shape[0]/256)+1)*256-b.shape[0]),(0,(int(b.shape[1]/256)+1)*256-b.shape[1])),  'constant', constant_values=(0,0))
#    image = np.dstack((r,g,b))
#    image=Image.fromarray(image)
#    x,y = image.size
#    imglist=cut_image(image)
#    list_out=[]
#    for img in imglist:        
#        r,g,b= img.split()
#        bgr=Image.merge('RGB',(r,g,b)) 
##        bgr=Image.merge('RGB',(b,g,r))    
#        if self.transform is not None:
#          img= self.transform(bgr)
#        list_out.append(img)
#    return list_out,x,y
    return image,train_list[index]

  def __len__(self):
#    train_list=glob.glob(self.root+'/*.bmp')
    train_list=glob.glob(self.root+'/*')
    return len(train_list)
