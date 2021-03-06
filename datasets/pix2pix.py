import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import glob
import scipy.ndimage
import colorsys 
import h5py
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
class pix2pix(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, seed=None):
    # imgs = make_dataset(root)
    # if len(imgs) == 0:
    #   raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
    #              "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root
    # self.imgs = imgs
    self.transform = transform
    self.loader = loader

    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):
    f = h5py.File(self.root+str(index)+'.hdf5','r')
    h=f['h'].value
    s=f['s'].value
    v = f['v'].value
    high = f['high'].value
    wide = f['wide'].value
    hsv = np.concatenate((h, s, v)).reshape(3,high,wide)/255.
#    print(index)
#    hsv = hsv.reshape(3,high,wide)
#    img=Image.open(self.root+str(index)+'.jpg')
#    print(index[1])
#    img_HSV = HSVColor(img)
    if self.transform is not None:
      img= self.transform(hsv)
    Light = img[2,:,:].mean()
#    Hue = img[0,:,:]
    return img,Light

  def __len__(self):
    train_list=glob.glob(self.root+'/*.hdf5')
    return len(train_list)
