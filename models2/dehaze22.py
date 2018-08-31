import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
import math


def Net(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s conv' % name, nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False))
  else:
    block.add_module('%s tconv' % name, nn.ConvTranspose2d(in_c, out_c, 3, 1, 1, bias=False))
  if bn:
    block.add_module('%s bn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%s dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block



class D(nn.Module):
  def __init__(self, input_nc, output_nc, ndf, n_layers):
    super(D, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s conv' % name, nn.Conv2d(input_nc+output_nc, ndf, 4, 2, 1, bias=False))
    main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    nf_mult = 1
    for n in range(1,n_layers+1):
        layer_idx= layer_idx+1
        name = 'layer%d' % layer_idx
        nf_mult_prev = nf_mult
        nf_mult = min(2**n,8)
        main.add_module('%s conv' % name, nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult,4,2,1))
        main.add_module('%s bn' % name, nn.BatchNorm2d(ndf * nf_mult))
        main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        
    layer_idx= layer_idx+1
    name = 'layer%d' % layer_idx
    nf_mult_prev = nf_mult
    nf_mult = min(2**n_layers,8)
    main.add_module('%s conv' % name, nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, 4, 1, 1))
    main.add_module('%s bn' % name, nn.BatchNorm2d(ndf * nf_mult))
    main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('F conv', nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1))       
    main.add_module('S sigmoid',nn.Sigmoid())

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output





class G(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(G, self).__init__()
    # input is 256 x 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = nn.Sequential()
    layer1.add_module(name, nn.Conv2d(input_nc, nf, 3, 1, 1, bias=False))
    # input is 128 x 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer2 = Net(nf, nf, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 64 x 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer3 = Net(nf, nf, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer4 = Net(nf, nf, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 16
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer5 = Net(nf, nf/2, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 8
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer6 = Net(nf/2, 1, name, transposed=False, bn=True, relu=False, dropout=False)

    ## NOTE: decoder
    # input is 1
    name = 'dlayer%d' % layer_idx
    dlayer6 = Net(1, nf/2, name, transposed=True, bn=False, relu=False, dropout=False)
    # input is 2
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer5 = Net(nf/2, nf, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 4
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer4 = Net(nf, nf, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 8
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer3 = Net(nf, nf, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 16
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer2 = Net(nf, nf, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 64
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer1 = nn.Sequential()
    dlayer1.add_module('%s relu' % name, nn.ReLU(inplace=True))
    dlayer1.add_module('%s tconv' % name, nn.ConvTranspose2d(nf, output_nc, 3, 1, 1, bias=False))

    dlayerfinal = nn.Sequential()
    dlayerfinal.add_module('%s tanh' % name, nn.Tanh())
    

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5
    self.layer6 = layer6
    self.dlayer6 = dlayer6
    self.dlayer5 = dlayer5
    self.dlayer4 = dlayer4
    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1
    self.dlayerfinal = dlayerfinal

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    dout1 = self.dlayer6(out6)
    dout2 = self.dlayer5(dout1)
    dout2 = dout2+out4
    dout3 = self.dlayer4(dout2)
    dout4 = self.dlayer3(dout3)
    dout4 = dout4+out2
    dout5 = self.dlayer2(dout4)
    dout6 = self.dlayer1(dout5)
    doutfinal = self.dlayerfinal(dout6)

    return doutfinal


