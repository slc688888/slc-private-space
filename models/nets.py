import torch.nn as nn
from torch.nn import functional as F
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
    main.add_module('%s conv' % name, nn.Conv2d(input_nc, ndf, 3, 1, 1, bias=False))
    main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    nf_mult = 1
    for n in range(1,n_layers+1):
        layer_idx= layer_idx+1
        name = 'layer%d' % layer_idx
        nf_mult_prev = nf_mult
        nf_mult = min(2**(n+1),8)
        main.add_module('%s conv' % name, nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult,5,2,1,bias=False))
        main.add_module('%s in' % name, nn.InstanceNorm2d(ndf * nf_mult))
        main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        
    layer_idx= layer_idx+1
    name = 'layer%d' % layer_idx
    nf_mult_prev = nf_mult
    nf_mult = min(2**n_layers,8)
    main.add_module('%s conv' % name, nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, 5, 2, 1))
    main.add_module('%s in' % name, nn.InstanceNorm2d(ndf * nf_mult))
    main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    
    layer_idx= layer_idx+1
    name = 'layer%d' % layer_idx
    nf_mult_prev = nf_mult
    nf_mult = min(2**n_layers,8)
    main.add_module('%s conv' % name, nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, 5, 2, 1))
    main.add_module('%s in' % name, nn.InstanceNorm2d(ndf * nf_mult))
    main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    
    main.add_module('F conv', nn.Conv2d(ndf * nf_mult, 16, 1, 1))       
    main.add_module('S sigmoid',nn.Sigmoid())

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output





class G(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(G, self).__init__()

    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = nn.Sequential()
    layer1.add_module(name, nn.Conv2d(input_nc, nf, 3, 1, 1, bias=False))#nf=16
    layer1.add_module('%s selu' % name, nn.selu())
    layer1.add_module('%s bn' % name, nn.BatchNorm2d(nf))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer2 = nn.Sequential()
    layer2.add_module(name, nn.Conv2d(nf, 2*nf, 5, 2, 1, bias=False))#nf=16
    layer2.add_module('%s selu' % name, nn.selu())
    layer2.add_module('%s bn' % name, nn.BatchNorm2d(2*nf))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer3 = nn.Sequential()
    layer3.add_module(name, nn.Conv2d(2*nf, 4*nf, 5, 2, 1, bias=False))#nf=16
    layer3.add_module('%s selu' % name, nn.selu())
    layer3.add_module('%s bn' % name, nn.BatchNorm2d(4*nf))    

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer4 = nn.Sequential()
    layer4.add_module(name, nn.Conv2d(4*nf, 8*nf, 5, 2, 1, bias=False))#nf=16
    layer4.add_module('%s selu' % name, nn.selu())
    layer4.add_module('%s bn' % name, nn.BatchNorm2d(8*nf))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer5 = nn.Sequential()
    layer5.add_module(name, nn.Conv2d(8*nf, 8*nf, 5, 2, 1, bias=False))#nf=16
    layer5.add_module('%s selu' % name, nn.selu())
    layer5.add_module('%s bn' % name, nn.BatchNorm2d(8*nf))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer6 = nn.Sequential()
    layer6.add_module(name, nn.Conv2d(8*nf, 8*nf, 5, 2, 1, bias=False))#nf=16
    layer6.add_module('%s selu' % name, nn.selu())
    layer6.add_module('%s bn' % name, nn.BatchNorm2d(8*nf))
    
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer7 = nn.Sequential()
    layer7.add_module(name, nn.Conv2d(8*nf, 8*nf, 5, 2, 1, bias=False))#nf=16
    layer7.add_module('%s selu' % name, nn.selu())
    layer7.add_module('%s bn' % name, nn.BatchNorm2d(8*nf))    
    
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer8 = nn.Sequential()
    layer8.add_module(name, nn.Conv2d(8*nf, 8*nf, 8, 1, 1, bias=False))#nf=16
    layer8.add_module('%s selu' % name, nn.selu())

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer9 = nn.Sequential()
    layer9.add_module(name, nn.Conv2d(8*nf, 8*nf, 1, 1, 1, bias=False))#nf=16
    
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer10 = nn.Sequential()
    layer10.add_module(name, nn.Conv2d(8*nf, 8*nf, 3, 1, 1, bias=False))#nf=16
    layer10.add_module('%s selu' % name, nn.selu())
    layer10.add_module('%s bn' % name, nn.BatchNorm2d(8*nf))       

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer11 = nn.Sequential()
    layer11.add_module(name, nn.Conv2d(8*nf, 8*nf, 1, 1, 1, bias=False))#nf=16
    layer11.add_module('%s selu' % name, nn.selu())
    layer11.add_module('%s bn' % name, nn.BatchNorm2d(8*nf))       
    
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer12 = nn.Sequential()
    layer12.add_module(name, nn.Conv2d(8*nf, 8*nf, 3, 1, 1, bias=False))#nf=16
    layer12.add_module('%s selu' % name, nn.selu())
    layer12.add_module('%s bn' % name, nn.BatchNorm2d(8*nf))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer13 = nn.Sequential()
    layer13.add_module(name, nn.Conv2d(8*nf, 8*nf, 3, 1, 1, bias=False))#nf=16
    layer13.add_module('%s selu' % name, nn.selu())
    layer13.add_module('%s bn' % name, nn.BatchNorm2d(8*nf))
    
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer14 = nn.Sequential()
    layer14.add_module(name, nn.Conv2d(8*nf, 4*nf, 3, 1, 1, bias=False))#nf=16
    layer14.add_module('%s selu' % name, nn.selu())
    layer14.add_module('%s bn' % name, nn.BatchNorm2d(4*nf))
    
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer15 = nn.Sequential()
    layer15.add_module(name, nn.Conv2d(4*nf, 2*nf, 3, 1, 1, bias=False))#nf=16
    layer15.add_module('%s selu' % name, nn.selu())
    layer15.add_module('%s bn' % name, nn.BatchNorm2d(2*nf))    

        
           
    ## NOTE: decoder
    # input is 1
    name = 'dlayer%d' % layer_idx
    dlayer6 = Net(1, int(nf/2), name, transposed=True, bn=False, relu=False, dropout=False)
    # input is 2
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer5 = Net(int(nf/2), nf, name, transposed=True, bn=True, relu=True, dropout=False)
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
    out1 = self.layer1(x) #16
    out2 = self.layer2(out1) #32
    out3 = self.layer3(out2) #64
    out4 = self.layer4(out3) #128
    out5 = self.layer5(out4)#128
    out6 = self.layer6(out5)#h/32
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


