import torch.nn as nn
from torch.nn import functional as F
import torch
import functools
class selu(nn.Module):
    def __init__(self):
        super(selu, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
    def forward(self, x):
        temp1 = self.scale * F.relu(x)
        temp2 = self.scale * self.alpha * (F.elu(-1*F.relu(-1*x)))
        return temp1 + temp2

def Net(in_c, out_c, name, transposed=False, bn=False, selu=True, dropout=False):
  block = nn.Sequential()
  if not transposed:
    block.add_module('%s conv' % name, nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False))
  else:
    block.add_module('%s tconv' % name, nn.ConvTranspose2d(in_c, out_c, 3, 1, 1, bias=False))
  if selu:
    block.add_module('%s selu' % name, nn.selu())
  else:
    block.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
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
    main.add_module('%s conv' % name, nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False))
    main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    nf_mult = 1
    for n in range(1,n_layers):
        layer_idx= layer_idx+1
        name = 'layer%d' % layer_idx
        nf_mult_prev = nf_mult
        nf_mult = min(2**n,8)
        main.add_module('%s conv' % name, nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult,4,2,1,bias=False))
        main.add_module('%s in' % name, nn.InstanceNorm2d(ndf * nf_mult))
        main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        
    layer_idx= layer_idx+1
    name = 'layer%d' % layer_idx
    nf_mult_prev = nf_mult
    nf_mult = min(2**n_layers,8)
    main.add_module('%s conv' % name, nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, 4, 1, 1))
    main.add_module('%s in' % name, nn.InstanceNorm2d(ndf * nf_mult))
    main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
#    
    layer_idx= layer_idx+1
    name = 'layer%d' % layer_idx
    nf_mult_prev = nf_mult
    nf_mult = min(2**n_layers,8)
    main.add_module('%s conv' % name, nn.Conv2d(ndf*nf_mult_prev, 1, 4, 1, 1))
#    main.add_module('%s in' % name, nn.InstanceNorm2d(ndf * nf_mult))
#    main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    
#    main.add_module('F conv', nn.Conv2d(ndf * nf_mult, 1, 4, 1))       
    main.add_module('S sigmoid',nn.Sigmoid())

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output

def doubleconv(in_c, out_c):
    block = nn.Sequential()
    block.add_module('conv', nn.Conv2d(in_c, out_c, 3,padding=1))
    block.add_module('bn', nn.BatchNorm2d(out_c))
    block.add_module('relu', nn.ReLU(inplace=True))
#    block.add_module('leakyrelu', nn.LeakyReLU(0.2, inplace=True))
    block.add_module('conv', nn.Conv2d(in_c, out_c, 3,padding=1))
    block.add_module('bn', nn.BatchNorm2d(out_c))
    block.add_module('relu', nn.ReLU(inplace=True))
#    block.add_module('leakyrelu', nn.LeakyReLU(0.2, inplace=True))
    return block    

#class DoubleConv(nn.Module):
#    def __init__(self, in_ch, out_ch):
#        super(DoubleConv, self).__init__()
#        self.conv = nn.Sequential(
#            nn.Conv2d(in_ch, out_ch, 3, padding=1),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(out_ch, out_ch, 3, padding=1),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True)
#        )
#
#    def forward(self, input):
#        return self.conv(input)

class G(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(G, self).__init__()
        self.conv1 = doubleconv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = doubleconv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = doubleconv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = doubleconv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = doubleconv(512, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = doubleconv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = doubleconv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = doubleconv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = doubleconv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return c10

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
        

class D2(nn.Module):
  def __init__(self, nc, nf):
    super(D2, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    main.add_module('%s bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%s sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output


def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
  else:
    block.add_module('%s tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
  if bn:
    block.add_module('%s bn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%s dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block