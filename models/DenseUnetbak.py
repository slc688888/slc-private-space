import torch.nn as nn
from torch.nn import functional as F
import torch
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
#        main.add_module('%s dropout' % name, nn.Dropout2d(0.5, inplace=True))
        main.add_module('%s leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
#        main.add_module('%s dropout' % name, nn.Dropout2d(0.5, inplace=True))
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
#    main.add_module('%s linear' % name, nn.linear(ndf*nf_mult_prev, 1, 4, 1, 1))
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


class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False,padding=0)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False,padding=0))
#        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2,padding=0))
        self.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2,padding=0))
        
class DTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(DTransition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False,padding=0))
        self.add_module('dconv', nn.ConvTranspose2d(num_output_features,num_output_features,kernel_size=2, stride=2,padding=0))
class G(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),num_init_features=32, bn_size=4, drop_rate=0, num_classes=1000):
        super(G, self).__init__()
        
        init_net = nn.Sequential()
        init_net.add_module('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
        init_net.add_module('norm0', nn.BatchNorm2d(num_init_features))
        init_net.add_module('relu0', nn.ReLU(inplace=True))
#        init_net.add_module('pool0', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))#256
        init_net.add_module('pool0', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))#256
        self.init_layer=init_net

        num_features = num_init_features
        num_layers=9
        self.block1 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.trans1 = Transition(num_input_features=num_features, num_output_features= num_init_features * 2)#128
        num_features = num_init_features * 2
        self.block2 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.trans2 = Transition(num_input_features=num_features, num_output_features= num_init_features * 4) #64
        num_features = num_init_features * 4
        self.block3 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.trans3 = Transition(num_input_features=num_features, num_output_features= num_init_features * 8)#32
        num_features = num_init_features * 8
        self.block4 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.trans4 = Transition(num_input_features=num_features, num_output_features= num_init_features * 16)#16
        num_features = num_init_features * 16
        self.block5 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.trans5 = Transition(num_input_features=num_features, num_output_features= num_init_features * 32)#8
        
        num_features = num_init_features * 32
        self.dblock1 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans1 = DTransition(num_input_features=num_features, num_output_features= num_init_features * 16)#512,1024
        num_features = num_init_features * 32
        self.dblock2 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans2 = DTransition(num_input_features=num_features, num_output_features= num_init_features * 8)#256,1024 
        num_features = num_init_features *16
        self.dblock3 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans3 = DTransition(num_input_features=num_features, num_output_features= num_init_features * 4)#128,512
        num_features = num_init_features * 8
        self.dblock4 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans4 = DTransition(num_input_features=num_features, num_output_features= num_init_features * 2)#64,256
        num_features = num_init_features * 4
        self.dblock5 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans5 = DTransition(num_input_features=num_features, num_output_features= num_init_features)
        
        final_layer = nn.Sequential()
        final_layer.add_module('dconv0', nn.ConvTranspose2d(num_init_features*2, 3, kernel_size=2, stride=2, padding=0, bias=False))
        self.final_layer = final_layer


    def forward(self, x):
        inputdata=self.init_layer(x)
        b1 = self.block1(inputdata)
        t1 = self.trans1(b1)
        b2 = self.block2(t1)
        t2 = self.trans2(b2)
        b3 = self.block3(t2)
        t3 = self.trans3(b3)
        b4 = self.block4(t3)
        t4 = self.trans4(b4)
        b5 = self.block5(t4)
        t5 = self.trans5(b5)
        d1 = self.dblock1(t5)
        dt1 = self.dtrans1(d1)
#        print(t5.size(),dt1.size())
#        dt1.resize_(t4.size(0),t4.size(1),t4.size(2),t4.size(3))
        merge1 = torch.cat([t4, dt1], dim=1)
        d2 = self.dblock2(merge1)
        dt2 = self.dtrans2(d2)
#        dt2.resize_(t3.size(0),t3.size(1),t3.size(2),t3.size(3))
        merge2 = torch.cat([t3, dt2], dim=1)  
        d3 = self.dblock3(merge2)
        dt3 = self.dtrans3(d3)
#        dt3.resize_(t2.size(0),t2.size(1),t2.size(2),t2.size(3))
        merge3 = torch.cat([t2, dt3], dim=1)  
        d4 = self.dblock4(merge3)
        dt4 = self.dtrans4(d4)
#        dt4.resize_(t1.size(0),t1.size(1),t1.size(2),t1.size(3))
        merge4 = torch.cat([t1, dt4], dim=1)
        d5 = self.dblock5(merge4)
        dt5 = self.dtrans5(d5)
#        dt5.resize_(inputdata.size(0),inputdata.size(1),inputdata.size(2),inputdata.size(3))
        merge5 = torch.cat([inputdata, dt5], dim=1)
        output=self.final_layer(merge5)
        return output


        
class G_U(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(G_U, self).__init__()  
        init_net = nn.Sequential()
        init_net.add_module('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
        init_net.add_module('norm0', nn.BatchNorm2d(num_init_features))
        init_net.add_module('relu0', nn.ReLU(inplace=True))
        init_net.add_module('pool0', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))#256
        self.init_layer=init_net
        num_layers=9
        self.block0 = DenseBlock(num_layers, num_input_features=3,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 3 + num_layers * growth_rate
        self.trans0 = Transition(num_input_features=num_features, num_output_features= num_init_features)#64
        num_features = num_init_features
        self.block1 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.trans1 = Transition(num_input_features=num_features, num_output_features= num_init_features * 2)#128
        num_features = num_init_features * 2
        self.block2 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.trans2 = Transition(num_input_features=num_features, num_output_features= num_init_features * 4) #256
        num_features = num_init_features * 4
        self.block3 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.trans3 = Transition(num_input_features=num_features, num_output_features= num_init_features * 8)#512
        num_features = num_init_features * 8
        self.block4 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.trans4 = Transition(num_input_features=num_features, num_output_features= num_init_features * 16)#1024
        num_features = num_init_features * 16
        
        self.dblock1 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans1 = DTransition(num_input_features=num_features, num_output_features= num_init_features * 8)#512,1024
        num_features = num_init_features * 16
        self.dblock2 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans2 = DTransition(num_input_features=num_features, num_output_features= num_init_features * 4)#256,1024 
        num_features = num_init_features * 12
        self.dblock3 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans3 = DTransition(num_input_features=num_features, num_output_features= num_init_features * 2)#128,512
        num_features = num_init_features * 8
        self.dblock4 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans4 = DTransition(num_input_features=num_features, num_output_features= num_init_features)#64,256
        num_features = num_init_features * 4
        self.dblock5 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans5 = DTransition(num_input_features=num_features, num_output_features= 3)       
        
        num_features=128
        self.dblock11 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans11 = DTransition(num_input_features=num_features, num_output_features= 64)        
        self.dblock12 = DenseBlock(num_layers, num_input_features=128,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 128 + num_layers * growth_rate
        self.dtrans12 = DTransition(num_input_features=num_features, num_output_features= 3)   

        num_features=256
        self.dblock21 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans21 = DTransition(num_input_features=num_features, num_output_features= 128) 
        num_features=256
        self.dblock22 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans22 = DTransition(num_input_features=num_features, num_output_features= 64) 
        self.dblock23 = DenseBlock(num_layers, num_input_features=128,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 128 + num_layers * growth_rate
        self.dtrans23 = DTransition(num_input_features=num_features, num_output_features= 3)
        
        num_features=512
        self.dblock31 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans31 = DTransition(num_input_features=num_features, num_output_features= 256)           
        num_features=512
        self.dblock32 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans32 = DTransition(num_input_features=num_features, num_output_features= 128) 
        num_features=384
        self.dblock33 = DenseBlock(num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.dtrans33 = DTransition(num_input_features=num_features, num_output_features= 64)         
        self.dblock34 = DenseBlock(num_layers, num_input_features=192,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 192 + num_layers * growth_rate
        self.dtrans34 = DTransition(num_input_features=num_features, num_output_features= 3)            
    def forward(self, x):
        b0=self.block0(x)
        t0=self.trans0(b0)
        b1 = self.block1(t0)
        t1 = self.trans1(b1)
        b2 = self.block2(t1)
        t2 = self.trans2(b2)
        b3 = self.block3(t2)
        t3 = self.trans3(b3)
        b4 = self.block4(t3)
        t4 = self.trans4(b4)
        
        
#        b11=self.dblock11(t1)#128
#        t11=self.dtrans11(b11)
#        merge11=torch.cat([t0, t11], dim=1)
#        b12 = self.dblock12(merge11)#128
#        output1 = self.dtrans12(b12)#3   
  
        b21=self.dblock21(t2)#256
        t21=self.dtrans21(b21)
        merge21=torch.cat([t1, t21], dim=1)
        b22 = self.dblock22(merge21)#256
        t22 = self.dtrans22(b22)#64
        merge22=torch.cat([t0, t22], dim=1)
        b23 = self.dblock23(merge22)#128
        output2 = self.dtrans23(b23)
        
        b31 = self.dblock31(t3)#512
        t31 = self.dtrans31(b31)
        merge31=torch.cat([t2, t31], dim=1)
        b32 = self.dblock32(merge31)#512
        t32 = self.dtrans32(b32)    
        merge32=torch.cat([t1, t32], dim=1)
        merge32=torch.cat([t21, merge32], dim=1)
        b33 = self.dblock33(merge32)#512
        t33 = self.dtrans33(b33)#64
        merge33=torch.cat([t0, t33], dim=1)
        merge33=torch.cat([t22, merge33], dim=1)
        b34 = self.dblock34(merge33)#256
        output3 = self.dtrans34(b34)        
        
        d1 = self.dblock1(t4)
        dt1 = self.dtrans1(d1)
        merge1 = torch.cat([t3, dt1], dim=1)
        d2 = self.dblock2(merge1)
        dt2 = self.dtrans2(d2)
#        merge2 = torch.cat([t2, dt2], dim=1)  
        merge2 = torch.cat([merge31, dt2], dim=1)
        d3 = self.dblock3(merge2)
        dt3 = self.dtrans3(d3)
#        merge3 = torch.cat([t1, dt3], dim=1)
        merge3 = torch.cat([merge32, dt3], dim=1)
#        merge3 = torch.cat([t21, merge3], dim=1)
        d4 = self.dblock4(merge3)
        dt4 = self.dtrans4(d4)
#        merge4 = torch.cat([t0, dt4], dim=1)
        merge4 = torch.cat([merge33, dt4], dim=1)
#        merge4 = torch.cat([merge22, merge4], dim=1)
        d5 = self.dblock5(merge4)
        output4 = self.dtrans5(d5)
        return output4,output3,output2#,output1
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