import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from util.GlobalAttention import GlobalAttentionGeneral as ATT_NET

import functools
from .cin import CINorm2d
from .cbn import CBNorm2d

def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        c_norm_layer = functools.partial(CBNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        c_norm_layer = functools.partial(CINorm2d, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer, c_norm_layer

def get_nl_layer(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'sigmoid':
        nl_layer = nn.Sigmoid
    elif layer_type == 'tanh':
        nl_layer = nn.Tanh
    else:
        raise NotImplementedError('nl_layer layer [%s] is not found' % layer_type)
    return nl_layer    

def weights_init(init_type='xavier'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
    return init_fun    
    
class Conv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, pad_type='reflect', bias=True, norm_layer=None, nl_layer=None):
        super(Conv2dBlock, self).__init__()
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=0, bias=bias)
        if norm_layer is not None:
            self.norm = norm_layer(out_planes)
        else:
            self.norm = lambda x: x
        
        if nl_layer is not None:
            self.activation = nl_layer()
        else:
            self.activation = lambda x: x
                     
    def forward(self, x):
        return self.activation(self.norm(self.conv(self.pad(x))))
    
def conv3x3(in_planes, out_planes, norm_layer=None, nl_layer=None):
    "3x3 convolution with padding"
    return Conv2dBlock(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, pad_type='reflect', bias=False, norm_layer=norm_layer, nl_layer=nl_layer)                     

def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)
    
def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)
    


################ G networks ###################       
class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf, output_nc):
        super(GET_IMAGE_G, self).__init__()
        self.img = conv3x3(ngf, output_nc, nl_layer=nn.Tanh)

    def forward(self, h_code):
        return self.img(h_code)
        
class CResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, h_dim, c_dim, c_norm_layer=None, nl_layer=None):
        super(CResidualBlock, self).__init__()
        self.c1 = Conv2dBlock(h_dim,h_dim, kernel_size=3, stride=1, padding=1, pad_type='reflect', bias=False)
        self.n1 = c_norm_layer(h_dim, num_con=c_dim)
        self.l1 = nl_layer()
        self.c2 = Conv2dBlock(h_dim,h_dim, kernel_size=3, stride=1, padding=1, pad_type='reflect', bias=False)
        self.n2 = c_norm_layer(h_dim, num_con=c_dim)

    def forward(self, input):
        x, w, c = input[0], input[1], input[2]
        y = self.l1(self.n1(self.c1(x),c))
        y = self.n2(self.c2(y),c)
        if w is not None:
            y = y*w
        return [x + y, w, c] 
        
class TRBlock(nn.Module):
    def __init__(self, ngf, ngf_out, nc, ne=0, norm_layer=None, c_norm_layer=None, nl_layer=None, block_num=2,isUp=True):
        super(TRBlock, self).__init__()
        block = []
        for i in range(block_num):
            block.append(CResidualBlock(ngf, nc+ne, c_norm_layer=c_norm_layer,nl_layer=nl_layer))
        self.translate =  nn.Sequential(*block)
        if isUp:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                Conv2dBlock(ngf,ngf_out,kernel_size=3, stride=1, padding=1, pad_type='reflect', bias=False,norm_layer=norm_layer,nl_layer=nl_layer)
                )
            self.att = ATT_NET(ngf, nc)
        else:
            self.upsample = lambda x_i : x_i
            self.att = lambda x_i1, x_i2, x_i3 : [None, None]
            
    def getAttnMap(self):
        b,w = self.map.size(0),self.map.size(3)
        map = self.map.max(dim=1, keepdim=True)[0]
        map = torch.cat([map,self.map],dim=1)
        c = map.size(1)
        max = map.view(b,c,-1).max(dim=2, keepdim=True)[0].view(b,c,1,1)
        min = map.view(b,c,-1).min(dim=2, keepdim=True)[0].view(b,c,1,1)
        map = (map - min) /(max - min) * 2 -1
        map =  torch.transpose ( torch.transpose(map,2,3).contiguous().view(b,1,w*c,-1),2,3).contiguous().view(b,1,-1,w*c)
        return map.repeat(1,3,1,1)
        
    def forward(self, h_code, c_glob, c_local=None, mask=None):
        weight, self.map = self.att(h_code, c_local, mask)
        trOut = self.translate([h_code,weight,c_glob])
        out_code = self.upsample(trOut[0])
        return out_code
        

class RAG_NET(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, nc=256, ne=0, norm_type='instance'):
        super(RAG_NET, self).__init__()
        
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type)
        nl_layer = get_nl_layer(layer_type='relu')
        
        self.c1 = Conv2dBlock(input_nc, ngf, kernel_size=7, stride=1, padding=3, pad_type='reflect', bias=False)
        self.n1 = c_norm_layer(ngf,nc+ne)
        self.a1 = nl_layer()
        
        
        self.c2 = Conv2dBlock(ngf, ngf*2, kernel_size=4, stride=2, padding=1, pad_type='reflect', bias=False)
        self.n2 = c_norm_layer(ngf*2,nc+ne)
        self.a2 = nl_layer()
        
        self.c3 = Conv2dBlock(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1,  pad_type='reflect', bias=False)
        self.n3 = c_norm_layer(ngf*4,nc+ne)
        self.a3 = nl_layer()
        
        self.tr1 = TRBlock(ngf*4, ngf*4, nc, norm_layer=norm_layer, c_norm_layer=c_norm_layer, nl_layer=nl_layer, block_num=4, isUp=False, ne=ne)        
        self.img_net1 = GET_IMAGE_G(ngf*4,output_nc)
       
        self.tr2 = TRBlock(ngf*4, ngf*2, nc, norm_layer=norm_layer, c_norm_layer=c_norm_layer, nl_layer=nl_layer, block_num=1, isUp=True,ne=ne)      
        self.img_net2 = GET_IMAGE_G(ngf*2,output_nc)
       
        self.tr3 = TRBlock(ngf*2, ngf, nc, norm_layer=norm_layer, c_norm_layer=c_norm_layer, nl_layer=nl_layer, block_num=1, isUp=True,ne=ne)     
        self.img_net3 = GET_IMAGE_G(ngf,output_nc)
        if ne>0:
            self.cat = torch.cat
        else:
            self.cat = lambda x_1, x_2 : x_1[0]
        
    def getAttnMap(self):
        return [self.tr2.getAttnMap(), self.tr3.getAttnMap()]
    
    def forward(self, x, c_global, c_local, mask=None, e_global=None):
        fake_imgs = []
        c_global = self.cat([c_global,e_global],1)# cat multiple conditional informations for global image generation
        h1 = self.a1(self.n1(self.c1(x),c_global))
        h2 = self.a2(self.n2(self.c2(h1),c_global))
        h3 = self.a3(self.n3(self.c3(h2),c_global))
        h_code = self.tr1(h3, c_global)
        fake_img1 = self.img_net1(h_code)
        fake_imgs.append(fake_img1)
        h_code = self.tr2(h_code, c_global, c_local,mask)
        fake_img2 = self.img_net2(h_code)
        fake_imgs.append(fake_img2)
        h_code = self.tr3(h_code, c_global, c_local, mask)
        fake_img3 = self.img_net3(h_code)
        fake_imgs.append(fake_img3)

        return fake_imgs     
        

################ D networks ##########################
class D_NET(nn.Module):
    def __init__(self, input_nc=3, ndf=32, block_num=3, nc=256, use_con=True, use_sigmoid=False,  norm_type='instance'):
        super(D_NET, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type)
        nl_layer = get_nl_layer('lrelu')
        block = [Conv2dBlock(input_nc, ndf, kernel_size=4,stride=2,padding=1,bias=False,nl_layer=nl_layer)]
        dim_in=ndf
        for n in range(1, block_num):
            dim_out = min(dim_in*2, ndf*8)
            block += [Conv2dBlock(dim_in, dim_out, kernel_size=4, stride=2, padding=1,bias=False,norm_layer=norm_layer,nl_layer=nl_layer)]
            dim_in = dim_out
        dim_out = min(dim_in*2, ndf*8)
        self.conv = nn.Sequential(*block)
        
        self.pre =  Conv2dBlock(dim_in, 1, kernel_size=4, stride=1, padding=1,bias=True) 
        if use_con:
            self.c_con1 = Conv2dBlock(dim_in, nc, kernel_size=4, stride=1, padding=1, bias=False)
            self.c_norm1 = c_norm_layer(nc, nc)
            self.c_nl = nl_layer()
            self.c_con2 = Conv2dBlock(nc, 1, kernel_size=4, stride=1, padding=1,bias=True)
            if use_sigmoid:
                self.c_nl2 = nn.Sigmoid()
            else:
                self.c_nl2 = lambda x: x
        
    def forward(self, x, c=None):
        c_pre = None
        x_code = self.conv(x)  #
        pre = self.pre(x_code)
        if c is not None:
            c_pre = self.c_nl2(self.c_con2(self.c_nl(self.c_norm1(self.c_con1(x_code),c))))

        return pre, c_pre, x_code
        
    def c_pre(self,x_code,c):
        c_pre = self.c_nl2(self.c_con2(self.c_nl(self.c_norm1(self.c_con1(x_code),c))))
        return c_pre

class D_NET_Multi(nn.Module):
    def __init__(self, input_nc=3, ndf=32, block_num=3, nc=256, use_con=True, use_sigmoid=False,norm_type='instance'):
        super(D_NET_Multi, self).__init__()
        self.model_1 = D_NET(input_nc=input_nc, ndf=ndf, block_num=block_num, nc=nc, use_con=use_con, use_sigmoid=use_sigmoid,norm_type=norm_type)
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.model_2 = D_NET(input_nc=input_nc, ndf=ndf//2, block_num=block_num, nc=nc, use_con=use_con, use_sigmoid=use_sigmoid,norm_type=norm_type)
        
    def forward(self, x, c=None):
        pre1, c_pre1, x_code1 = self.model_1(x,c)
        
        pre2, c_pre2, x_code2 = self.model_2(self.down(x),c)

        return [pre1,pre2], [c_pre1,c_pre2], [x_code1,x_code2]
        
    def c_pre(self,x_code,c):
        x_code1, x_code2 = x_code
        c_pre1 = self.model_1.c_pre(x_code1,c)
        c_pre2 = self.model_2.c_pre(x_code2,c)
        return [c_pre1,c_pre2]
        
################ E networks ##########################
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out
        
class E_ResNet_Local(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, nef=64, n_blocks=4, norm_type='instance'):
        # img 128*128 -> n_blocks=5 // img 256*256 -> n_blocks=6
        super(E_ResNet_Local, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type)
        max_ndf = 4
        nl_layer = get_nl_layer(layer_type='lrelu')
        conv_layers = [Conv2dBlock(input_nc, nef, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = nef * min(max_ndf, n)  # 2**(n-1)
            output_ndf = nef * min(max_ndf, n+1)  # 2**n
            conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer()]
        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
        
    def forward(self, x):
        x_conv = self.conv(x)
        b,c,h,w = x_conv.size(0), x_conv.size(1), x_conv.size(2), x_conv.size(3)
        x_local = torch.transpose(x_conv.view(b,c,-1),1,2).contiguous().view(-1,c)
        mu = self.fc(x_local)
        logvar = self.fcVar(x_local)
        c_code = self.reparametrize(mu, logvar).view(b,h*w,-1)
        c_code_glob = torch.mean(c_code,dim=1).view(b,-1)       #shape [b,c]
        c_code_local = torch.transpose(c_code,1,2).contiguous() #shape [b,c,h*w]
        return c_code_glob, c_code_local, mu, logvar

class E_ResNet_Global(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, nef=64, n_blocks=4, norm_type='instance'):
        # img 128*128 -> n_blocks=4 // img 256*256 -> n_blocks=5 
        super(E_ResNet_Global, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type)
        max_ndf = 4
        nl_layer = get_nl_layer(layer_type='lrelu')
        conv_layers = [Conv2dBlock(input_nc, nef, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = nef * min(max_ndf, n)  # 2**(n-1)
            output_ndf = nef * min(max_ndf, n+1)  # 2**n
            conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AdaptiveAvgPool2d(1)]
        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
        
    def forward(self, x):
        x_conv = self.conv(x)
        b = x_conv.size(0)
        x_conv = x_conv.view(b, -1)
        mu = self.fc(x_conv)
        logvar = self.fcVar(x_conv)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar 
        

class CN_BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, c_dim=1, c_norm_layer=None, nl_layer=None):
        super(CN_BasicBlock, self).__init__()
        self.norm1 = c_norm_layer(inplanes,c_dim)
        self.nl1 = nl_layer()
        self.c1 = conv3x3(inplanes, inplanes)
        self.norm2 = c_norm_layer(inplanes,c_dim)
        self.nl2 = nl_layer()
        self.c2 = convMeanpool(inplanes, outplanes)
        self.shortcut = meanpoolConv(inplanes, outplanes)
        
    def forward(self, x,z):
        out = self.c2(self.nl2(self.norm2(self.c1(self.nl1(self.norm1(x,z))),z))) + self.shortcut(x)
        return out        
        
class CE_ResNet_Local(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, nef=64, n_blocks=4, c_dim=2, norm_type='instance'):
        # img 128*128 -> n_blocks=5 // img 256*256 -> n_blocks=6 
        super(CE_ResNet_Local, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type)
        max_ndf = 4
        nl_layer = get_nl_layer(layer_type='lrelu')
        
        self.c1 = Conv2dBlock(input_nc, nef, kernel_size=4, stride=2, padding=1, bias=True)
        self.res1 = CN_BasicBlock(nef, nef*2, c_dim, c_norm_layer, nl_layer)
        self.res2 = CN_BasicBlock(nef*2, nef*3, c_dim, c_norm_layer, nl_layer)
        self.res3 = CN_BasicBlock(nef*3, nef*4, c_dim, c_norm_layer, nl_layer)
        self.res4 = CN_BasicBlock(nef*4, nef*4, c_dim, c_norm_layer, nl_layer)
        if n_blocks==6:
            self.res5 = CN_BasicBlock(nef*4, nef*4, c_dim, c_norm_layer, nl_layer)
        elif n_blocks ==5:
            self.res5 = lambda x_i,x_i2 : x_i
        self.nl_f = nl_layer()

        self.fc = nn.Sequential(*[nn.Linear(nef*4, output_nc)])
        self.fcVar = nn.Sequential(*[nn.Linear(nef*4, output_nc)])

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
        
    def forward(self, x, z):
        x_conv = self.nl_f(self.res5(self.res4(self.res3(self.res2(self.res1(self.c1(x),z),z),z),z),z))
        b,c,h,w = x_conv.size(0), x_conv.size(1), x_conv.size(2), x_conv.size(3)
        x_local = torch.transpose(x_conv.view(b,c,-1),1,2).contiguous().view(-1,c)
        mu = self.fc(x_local)
        logvar = self.fcVar(x_local)
        c_code = self.reparametrize(mu, logvar).view(b,h*w,-1)
        c_code_glob = torch.mean(c_code,dim=1).view(b,-1)       #shape [b,c]
        c_code_local = torch.transpose(c_code,1,2).contiguous() #shape [b,c,h*w]
        return c_code_glob, c_code_local, mu, logvar     
        
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = 18#cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = 'LSTM'#cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        emb = self.drop(self.encoder(captions))
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]

        words_emb = output.transpose(1, 2)
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


