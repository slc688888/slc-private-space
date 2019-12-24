import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import h5py
import glob
import pdb

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

class pix2pix_val(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, seed=None):
    # imgs = make_dataset(root)
    # if len(imgs) == 0:
    #   raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
    #              "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root
    # self.imgs = imgs
    self.transform = transform
    self.loader = loader
    # self.sampler = SequentialSampler(dataset)

    if seed is not None:
      np.random.seed(seed)

  # def __getitem__(self, _):
  def __getitem__(self, index):
    f = h5py.File(self.root+str(index)+'.hdf5','r')
    h=f['h'].value
    s=f['s'].value
    v = f['v'].value
    high = f['high'].value
    wide = f['wide'].value
    hsv = np.concatenate((h, s, v)).reshape(3,high,wide)/255.
    if self.transform is not None:
      img= self.transform(hsv)
    return img


  def __len__(self):
    train_list=glob.glob(self.root+'/*.hdf5')
    return len(train_list)

