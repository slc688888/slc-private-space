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
class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)
class HDC(nn.Module):
    def __init__(self, channel_num,d3=3,dilation=1, group=1):
        super(HDC, self).__init__()
#        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = nn.Conv2d(channel_num, 64, 3, 1, padding=1, dilation=1, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=d3, dilation=d3, groups=group, bias=False)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y = self.norm3(self.conv3(y))
        return F.relu(y)
class G_HDC(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(G_HDC, self).__init__()
        init_net = nn.Sequential()
        init_net.add_module('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
        init_net.add_module('norm0', nn.InstanceNorm2d(num_init_features,affine=True))
        init_net.add_module('relu0', nn.ReLU(inplace=True))
        self.init_layer=init_net

        first_layer = nn.Sequential()
        first_layer.add_module('conv1', nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
        first_layer.add_module('norm1', nn.InstanceNorm2d(num_init_features,affine=True))
        first_layer.add_module('relu1', nn.ReLU(inplace=True))
        self.first_layer=first_layer

        second_layer = nn.Sequential()
        second_layer.add_module('conv2', nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=2, padding=1, bias=False))
        second_layer.add_module('norm2', nn.InstanceNorm2d(num_init_features,affine=True))
        second_layer.add_module('relu2', nn.ReLU(inplace=True))
        self.second_layer=second_layer
      
        self.HDC1 = HDC(num_init_features)
        self.HDC11 = HDC(num_init_features)
        
        self.HDC2 = HDC(num_init_features*2)
        self.HDC22 = HDC(num_init_features)
        
        up_layer = nn.Sequential()
        up_layer.add_module('convup', nn.ConvTranspose2d(num_init_features*3, num_init_features, kernel_size=4, stride=2, padding=1))
        up_layer.add_module('normup', nn.InstanceNorm2d(num_init_features,affine=True))
        up_layer.add_module('reluup', nn.ReLU(inplace=True))
        self.up_layer=up_layer      
        
        self.HDC3 = HDC(num_init_features*3)
        self.HDC33 = HDC(num_init_features)
                
        last_layer = nn.Sequential()
        last_layer.add_module('convlast', nn.ConvTranspose2d(num_init_features*4, num_init_features, kernel_size=4, stride=2, padding=1))
        last_layer.add_module('normlast', nn.InstanceNorm2d(num_init_features,affine=True))
        last_layer.add_module('relulast', nn.ReLU(inplace=True))
        self.last_layer=last_layer

        final_layer = nn.Sequential()
        final_layer.add_module('conv', nn.Conv2d(num_init_features*3, num_init_features, kernel_size=3, stride=1, padding=1))
        final_layer.add_module('relu', nn.ReLU(inplace=True))
        self.final_layer=final_layer 
        
        final_net = nn.Sequential()
        final_net.add_module('convfinal', nn.Conv2d((num_init_features+3), 3, kernel_size=3, stride=1, padding=1))
        self.final_net=final_net  
    def forward(self, x): 
        """first version"""        
        y00=self.init_layer(x)
        y0=self.first_layer(y00)
        y1=self.second_layer(y0)
        
        HDC1=self.HDC1(y1)
        HDC1=self.HDC11(HDC1)
        y2=F.max_pool2d(torch.cat((HDC1, y1), dim=1),kernel_size=2, stride=2,padding=0)
        
        HDC2=self.HDC2(y2)
        HDC2=self.HDC22(HDC2)
        y3=torch.cat((HDC2, y2), dim=1)
        y3=self.up_layer(y3)
        y3=torch.cat((y3,y1,HDC1), dim=1)  
        
        HDC3=self.HDC3(y3)
        HDC3=self.HDC33(HDC3)
        y4=torch.cat((HDC3,y3), dim=1) 
        
        y4=self.last_layer(y4)
        y5=torch.cat((y4,y0,y00), dim=1)
        y5=self.final_layer(y5)
        y5=torch.cat((y5,x), dim=1)
        output=self.final_net(y5)      
        return F.relu(output)
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


class P(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),num_init_features=32, bn_size=4, drop_rate=0, num_classes=1000):
        super(P, self).__init__()
        
        init_net = nn.Sequential()
        init_net.add_module('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
        init_net.add_module('norm0', nn.BatchNorm2d(num_init_features))
        init_net.add_module('relu0', nn.ReLU(inplace=True))
#        init_net.add_module('pool0', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))#128
        self.init_layer=init_net

        num_features = num_init_features
        num_layers=3
        """vgg arc"""
        self.vggblock1 = DenseBlock(num_layers, num_input_features=32,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 32 + num_layers * growth_rate
        self.vggtrans1 = Transition(num_input_features=num_features, num_output_features= 64)#64
        
        self.vggblock2 = DenseBlock(num_layers, num_input_features=64,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 64 + num_layers * growth_rate
        self.vggtrans2 = Transition(num_input_features=num_features, num_output_features= 128)#32

        self.vggblock3 = DenseBlock(num_layers, num_input_features=128,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 128 + num_layers * growth_rate
        self.vggtrans3 = Transition(num_input_features=num_features, num_output_features= 256)#16
        
        self.vggblock4 = DenseBlock(num_layers, num_input_features=256,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 256 + num_layers * growth_rate
        self.vggtrans4 = Transition(num_input_features=num_features, num_output_features= 512)#8 

        self.vggblock5 = DenseBlock(num_layers, num_input_features=512,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 512 + num_layers * growth_rate
        self.vggtrans5 = Transition(num_input_features=num_features, num_output_features= 1024)#8 
        
        self.vggdblock1 = DenseBlock(num_layers, num_input_features=1024,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 1024 + num_layers * growth_rate
        self.vggdtrans1 = DTransition(num_input_features=num_features, num_output_features= 512)

        self.vggdblock2 = DenseBlock(num_layers, num_input_features=1024,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 1024 + num_layers * growth_rate
        self.vggdtrans2 = DTransition(num_input_features=num_features, num_output_features= 256)

        self.vggdblock3 = DenseBlock(num_layers, num_input_features=512,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 512 + num_layers * growth_rate
        self.vggdtrans3 = DTransition(num_input_features=num_features, num_output_features= 128)

        self.vggdblock4 = DenseBlock(num_layers, num_input_features=256,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 256 + num_layers * growth_rate
        self.vggdtrans4 = DTransition(num_input_features=num_features, num_output_features= 64)             

        self.vggdblock5 = DenseBlock(num_layers, num_input_features=128,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 128 + num_layers * growth_rate
        self.vggdtrans5 = DTransition(num_input_features=num_features, num_output_features= 32)    
        """hf arc"""
        self.hfblock1 = DenseBlock(num_layers, num_input_features=32,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 32 + num_layers * growth_rate
        self.hftrans1 = Transition(num_input_features=num_features, num_output_features= 64)#256channel  
        
        self.hfblock2 = DenseBlock(num_layers, num_input_features=64,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 64 + num_layers * growth_rate
        self.hftrans2 = Transition(num_input_features=num_features, num_output_features= 128)#256channel  
        
        self.hfblock3 = DenseBlock(num_layers, num_input_features=128,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 128 + num_layers * growth_rate
        self.hftrans3 = Transition(num_input_features=num_features, num_output_features= 256)#256channel  
        
        self.hfblock4 = DenseBlock(num_layers, num_input_features=256,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 256 + num_layers * growth_rate
        self.hftrans4 = Transition(num_input_features=num_features, num_output_features= 512)#256channel 

        self.hfblock5 = DenseBlock(num_layers, num_input_features=512,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 512 + num_layers * growth_rate
        self.hftrans5 = Transition(num_input_features=num_features, num_output_features= 1024)#256channel 
        
        self.hfdblock1 = DenseBlock(num_layers, num_input_features=1024,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 1024 + num_layers * growth_rate
        self.hfdtrans1 = DTransition(num_input_features=num_features, num_output_features= 512) 

        self.hfdblock2 = DenseBlock(num_layers, num_input_features=1024,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 1024 + num_layers * growth_rate
        self.hfdtrans2 = DTransition(num_input_features=num_features, num_output_features= 256) 

        self.hfdblock3 = DenseBlock(num_layers, num_input_features=512,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 512 + num_layers * growth_rate
        self.hfdtrans3 = DTransition(num_input_features=num_features, num_output_features= 128) 
        
        self.hfdblock4 = DenseBlock(num_layers, num_input_features=256,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 256 + num_layers * growth_rate
        self.hfdtrans4 = DTransition(num_input_features=num_features, num_output_features= 64)

        self.hfdblock5 = DenseBlock(num_layers, num_input_features=128,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 128 + num_layers * growth_rate
        self.hfdtrans5 = DTransition(num_input_features=num_features, num_output_features= 32)           
        """ff arc"""
        self.ffblock1 = DenseBlock(num_layers, num_input_features=32,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 32 + num_layers * growth_rate
        self.fftrans1 = Transition(num_input_features=num_features, num_output_features= 64)#256channel    

        self.ffblock2 = DenseBlock(num_layers, num_input_features=64,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 64 + num_layers * growth_rate
        self.fftrans2 = Transition(num_input_features=num_features, num_output_features= 128)#256channel   
        
        self.ffblock3 = DenseBlock(num_layers, num_input_features=128,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 128 + num_layers * growth_rate
        self.fftrans3 = Transition(num_input_features=num_features, num_output_features= 256)#256channel           
        
        self.ffblock4 = DenseBlock(num_layers, num_input_features=256,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 256 + num_layers * growth_rate
        self.fftrans4 = Transition(num_input_features=num_features, num_output_features= 512)#256channel 

        self.ffblock5 = DenseBlock(num_layers, num_input_features=512,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 512 + num_layers * growth_rate
        self.fftrans5 = Transition(num_input_features=num_features, num_output_features= 1024)#256channel 
          
        self.ffdblock1 = DenseBlock(num_layers, num_input_features=1024,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 1024 + num_layers * growth_rate
        self.ffdtrans1 = DTransition(num_input_features=num_features, num_output_features= 512) 

        self.ffdblock2 = DenseBlock(num_layers, num_input_features=1024,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 1024 + num_layers * growth_rate
        self.ffdtrans2 = DTransition(num_input_features=num_features, num_output_features= 256) 
        
        self.ffdblock3 = DenseBlock(num_layers, num_input_features=512,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 512 + num_layers * growth_rate
        self.ffdtrans3 = DTransition(num_input_features=num_features, num_output_features= 128)  
        
        self.ffdblock4 = DenseBlock(num_layers, num_input_features=256,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 256 + num_layers * growth_rate
        self.ffdtrans4 = DTransition(num_input_features=num_features, num_output_features= 64) 
        
        self.ffdblock5 = DenseBlock(num_layers, num_input_features=128,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = 128 + num_layers * growth_rate
        self.ffdtrans5 = DTransition(num_input_features=num_features, num_output_features= 32) 
        """final"""        
        final_layer = nn.Sequential()
        final_layer.add_module('dconv0', nn.ConvTranspose2d(128, 3, kernel_size=1, stride=1, padding=0, bias=False))
        self.final_layer = final_layer
        
        


    def forward(self, x):
        inputdata=self.init_layer(x)
        vgg1 = self.vggblock1(inputdata)
        vgg1 = self.vggtrans1(vgg1)
        vgg2 = self.vggblock2(vgg1)
        vgg2 = self.vggtrans2(vgg2)        
        vgg3 = self.vggblock3(vgg2)
        vgg3 = self.vggtrans3(vgg3) 
        vgg4 = self.vggblock4(vgg3)
        vgg4 = self.vggtrans4(vgg4)
        vgg5 = self.vggblock5(vgg4)
        vgg5 = self.vggtrans5(vgg5) 
         
        vggd1 = self.vggdblock1(vgg5)
        vggd1 = self.vggdtrans1(vggd1)
        vggd1 = torch.cat([vggd1, vgg4], dim=1)
        vggd2 = self.vggdblock2(vggd1)
        vggd2 = self.vggdtrans2(vggd2)
        vggd2 = torch.cat([vggd2, vgg3], dim=1)
        vggd3 = self.vggdblock3(vggd2)
        vggd3 = self.vggdtrans3(vggd3)
        vggd3 = torch.cat([vggd3, vgg2], dim=1)
        vggd4 = self.vggdblock4(vggd3)
        vggd4 = self.vggdtrans4(vggd4)    
        vggd4 = torch.cat([vggd4, vgg1], dim=1)
        vggd5 = self.vggdblock5(vggd4)
        vggd5 = self.vggdtrans5(vggd5) 
        
        hf1 = self.hfblock1(inputdata)
        hf1 = self.hftrans1(hf1)
        hf2 = self.hfblock2(hf1)
        hf2 = self.hftrans2(hf2)        
        hf3 = self.hfblock3(hf2)
        hf3 = self.hftrans3(hf3) 
        hf4 = self.hfblock4(hf3)
        hf4 = self.hftrans4(hf4)
        hf5 = self.hfblock5(hf4)
        hf5 = self.hftrans5(hf5)
         
        hfd1 = self.hfdblock1(hf5)
        hfd1 = self.hfdtrans1(hfd1)
        hfd1 = torch.cat([hfd1, hf4], dim=1)
        hfd2 = self.hfdblock2(hfd1)
        hfd2 = self.hfdtrans2(hfd2)
        hfd2 = torch.cat([hfd2, hf3], dim=1)
        hfd3 = self.hfdblock3(hfd2)
        hfd3 = self.hfdtrans3(hfd3)
        hfd3 = torch.cat([hfd3, hf2], dim=1)
        hfd4 = self.hfdblock4(hfd3)
        hfd4 = self.hfdtrans4(hfd4)         
        hfd4 = torch.cat([hfd4, hf1], dim=1)
        hfd5 = self.hfdblock5(hfd4)
        hfd5 = self.hfdtrans5(hfd5) 
        
        ff1 = self.ffblock1(inputdata)
        ff1 = self.fftrans1(ff1)
        ff2 = self.ffblock2(ff1)
        ff2 = self.fftrans2(ff2)        
        ff3 = self.ffblock3(ff2)
        ff3 = self.fftrans3(ff3) 
        ff4 = self.ffblock4(ff3)
        ff4 = self.fftrans4(ff4)
        ff5 = self.ffblock5(ff4)
        ff5 = self.fftrans5(ff5)
         
        ffd1 = self.ffdblock1(ff5)
        ffd1 = self.ffdtrans1(ffd1)
        ffd1 = torch.cat([ffd1, ff4], dim=1)
        ffd2 = self.ffdblock2(ffd1)
        ffd2 = self.ffdtrans2(ffd2)
        ffd2 = torch.cat([ffd2, ff3], dim=1)
        ffd3 = self.ffdblock3(ffd2)
        ffd3 = self.ffdtrans3(ffd3)
        ffd3 = torch.cat([ffd3, ff2], dim=1)
        ffd4 = self.ffdblock4(ffd3)
        ffd4 = self.ffdtrans4(ffd4) 
        ffd4 = torch.cat([ffd4, ff1], dim=1)
        ffd5 = self.ffdblock5(ffd4)
        ffd5 = self.ffdtrans5(ffd5)         
        vggok = torch.cat([vggd5, inputdata], dim=1)
        hfok = torch.cat([hfd5, ffd5], dim=1)
        ok = torch.cat([vggok, hfok], dim=1)
        output=self.final_layer(ok)
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


class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(SmoothDilatedResidualBlock, self).__init__()
        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
        y = self.norm2(self.conv2(self.pre_conv2(y)))
        return F.relu(x+y)


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)


class GCANet(nn.Module):
    def __init__(self, in_c=3, out_c=3, only_residual=True):
        super(GCANet, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 64, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)

        self.res1 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res2 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res3 = SmoothDilatedResidualBlock(64, dilation=2)
        self.res4 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res5 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res6 = SmoothDilatedResidualBlock(64, dilation=4)
        self.res7 = ResidualBlock(64, dilation=1)

        self.gate = nn.Conv2d(64 * 3, 3, 3, 1, 1, bias=True)

        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.norm4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.norm5 = nn.InstanceNorm2d(64, affine=True)
        self.deconv1 = nn.Conv2d(64, out_c, 1)
        self.only_residual = only_residual

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y1 = F.relu(self.norm3(self.conv3(y)))

        y = self.res1(y1)
        y = self.res2(y)
        y = self.res3(y)
        y2 = self.res4(y)
        y = self.res5(y2)
        y = self.res6(y)
        y3 = self.res7(y)

        gates = self.gate(torch.cat((y1, y2, y3), dim=1))
        gated_y = y1 * gates[:, [0], :, :] + y2 * gates[:, [1], :, :] + y3 * gates[:, [2], :, :]
        y = F.relu(self.norm4(self.deconv3(gated_y)))
        y = F.relu(self.norm5(self.deconv2(y)))
        if self.only_residual:
            y = self.deconv1(y)
        else:
            y = F.relu(self.deconv1(y))

        return y