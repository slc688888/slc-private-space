#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
from torch.autograd import Variable
#from yolov3.darknet import Darknet
import misc
import torchvision.utils as vutils
import models.DenseUnet as net
#from myutils.vgg16 import VGG
#from myutils import utils
import torchvision.models as models
from yolov3.models import *
import xml.etree.ElementTree as ET
import numpy as np
from yolov3.utils.utils import *
from yolov3.utils.datasets import *
from yolov3.utils.parse_config import *
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='test',  help='')
parser.add_argument('--dataroot', required=False,
  default='./RTTS/JPEGImages/', help='path to trn dataset')
parser.add_argument('--datarootfree', required=False,
  default='./D_HAZE/NYU_GT/', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='./facades/test256_new', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=512, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=512, help='the height / width of the cropped input image to network')
parser.add_argument('--condition_GAN', type=int,
  default=1, help='set to 0 to use unconditional discriminator')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=48)
parser.add_argument('--niter', type=int, default=80, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=40, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lamdaA', type=float, default=0.0066, help='lambdaGAN')
parser.add_argument('--lamdaP', type=float, default=1, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='./DUADN_256/netG_newepoch_78.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample_DN_yolo', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=1, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=50, help='interval for evauating(generating) images from valDataroot')

parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--model_def", type=str, default="yolov3/config/yolov3-custom.cfg", help="path to model definition file")
parser.add_argument("--data_config", type=str, default="yolov3/config/custom.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="yolov3/weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="yolov3/data/coco.names", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.005, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
parser.add_argument("--image_folder", type=str, default="yolov3/data/samples", help="path to dataset")
opt = parser.parse_args()
print(opt)
'''init netG,netD'''
netG = net.G()
if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
print(netG)
#netG.eval()
netG.cuda()

def evaluate(model,netG, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    netG.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []
    # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
#        targets=targets.float().cuda()

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            dehaze = netG(imgs)   
#            current_time = time.time()
#            inference_time = datetime.timedelta(seconds=current_time - prev_time)
#            prev_time = current_time
#            print(inference_time)
#            vutils.save_image(dehaze, 'test.png', normalize=False,scale_each=False)
            outputs = model(dehaze)
#            vutils.save_image(dehaze, 'test.png', normalize=False,scale_each=False)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_config = parse_data_config(opt.data_config)
valid_path = data_config["valid"]
class_names = load_classes(data_config["names"])

# Initiate model
model = Darknet(opt.model_def).to(device)
if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))
model.eval()
print("Compute mAP...")

precision, recall, AP, f1, ap_class = evaluate(
    model,
    netG,
    path=valid_path,
    iou_thres=opt.iou_thres,
    conf_thres=opt.conf_thres,
    nms_thres=opt.nms_thres,
    img_size=opt.img_size,
    batch_size=1,
)

print("Average Precisions:")
for i, c in enumerate(ap_class):
    print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

print(f"mAP: {AP.mean()}")
