from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image
from model import RAG_NET, D_NET, D_NET_Multi, RNN_ENCODER, E_ResNet_Global, E_ResNet_Local, weights_init, CE_ResNet_Local
from util.loss import GANLoss, KL_loss
import os
import time
import numpy as np
import sys

################## Residual Attention for image-to-image translation #############################
class RAGAN():
    def name(self):
        return 'ResidualAttentionGAN'

    def initialize(self, opt):
        torch.cuda.set_device(opt.gpu)
        cudnn.benchmark = True
        self.D_len = 3
        self.opt = opt
        self.build_models()
        
        
    def build_models(self):
        ################### encoders #########################################
        self.E_image = None
        self.E_text = None
        use_con = False
        use_sigmoid = True if self.opt.c_gan_mode == 'dcgan' else False
        if self.opt.c_type == 'image':
            if self.opt.model == 'supervised':
                self.E_image = E_ResNet_Local(input_nc=self.opt.output_nc, output_nc=self.opt.nc, nef=self.opt.nef, n_blocks=self.opt.e_blocks,norm_type=self.opt.norm)
            elif self.opt.model == 'unsupervised':
                self.E_image = CE_ResNet_Local(input_nc=self.opt.output_nc, output_nc=self.opt.nc, nef=self.opt.nef, n_blocks=self.opt.e_blocks, c_dim=self.opt.ne, norm_type=self.opt.norm)
        elif self.opt.c_type == 'text':
            use_con = True
            self.E_text =  RNN_ENCODER(self.opt.n_words, nhidden=self.opt.nc)
            state_dict = torch.load(self.opt.E_text_path, map_location=lambda storage, loc: storage)
            self.E_text.load_state_dict(state_dict)
            for p in self.E_text.parameters():
                p.requires_grad = False
            print('Load text encoder successful')
            self.E_text.eval()
        elif self.opt.c_type == 'image_text':
            use_con = True
            self.E_image = E_ResNet_Global(input_nc=self.opt.output_nc, output_nc=self.opt.ne, nef=self.opt.nef, n_blocks=self.opt.e_blocks,norm_type=self.opt.norm)
            self.E_text =  RNN_ENCODER(self.opt.n_words, nhidden=self.opt.nc)
            state_dict = torch.load(self.opt.E_text_path, map_location=lambda storage, loc: storage)
            self.E_text.load_state_dict(state_dict)
            for p in self.E_text.parameters():
                p.requires_grad = False
            print('Load text encoder successful')
            self.E_text.eval()
        elif self.opt.c_type == 'label':
            use_con = True
        elif self.opt.c_type == 'image_label':
            use_con = True
            self.E_image = E_ResNet_Global(input_nc=self.opt.output_nc, output_nc=self.opt.ne, nef=self.opt.nef, n_blocks=self.opt.e_blocks,norm_type=self.opt.norm)
        else:
            raise('Non conditioanl type of {}'.format(self.opt.c_type))
        
        
        ################### generator #########################################
        self.G = RAG_NET(input_nc=self.opt.input_nc, ngf=self.opt.ngf, nc=self.opt.nc, ne=self.opt.ne,norm_type=self.opt.norm)
        
        if self.opt.isTrain:    
            ################### discriminators #####################################
            self.Ds = []
            self.Ds2 = None
            bnf = 3 if self.opt.fineSize <=128 else 4
            self.Ds.append(D_NET(input_nc=self.opt.output_nc, ndf=self.opt.ndf, block_num=bnf, nc=self.opt.nc, use_con=use_con, use_sigmoid=use_sigmoid,norm_type=self.opt.norm))
            self.Ds.append(D_NET(input_nc=self.opt.output_nc, ndf=self.opt.ndf, block_num=4, nc=self.opt.nc, use_con=use_con, use_sigmoid=use_sigmoid,norm_type=self.opt.norm))
            self.Ds.append(D_NET_Multi(input_nc=self.opt.output_nc, ndf=self.opt.ndf, block_num=4, nc=self.opt.nc, use_con=use_con, use_sigmoid=use_sigmoid,norm_type=self.opt.norm))
            if self.opt.model == 'unsupervised' and self.opt.c_type == 'image':
                self.Ds2 = []
                self.Ds2.append(D_NET(input_nc=self.opt.output_nc, ndf=self.opt.ndf, block_num=bnf, nc=self.opt.nc, use_con=use_con, use_sigmoid=use_sigmoid,norm_type=self.opt.norm))
                self.Ds2.append(D_NET(input_nc=self.opt.output_nc, ndf=self.opt.ndf, block_num=4, nc=self.opt.nc, use_con=use_con, use_sigmoid=use_sigmoid,norm_type=self.opt.norm))
                self.Ds2.append(D_NET_Multi(input_nc=self.opt.output_nc, ndf=self.opt.ndf, block_num=4, nc=self.opt.nc, use_con=use_con, use_sigmoid=use_sigmoid,norm_type=self.opt.norm))
            ################### init_weights ########################################
            self.G.apply(weights_init(self.opt.init_type))
            for i in range(self.D_len):
                self.Ds[i].apply(weights_init(self.opt.init_type))
            if self.Ds2 is not None:
                for i in range(self.D_len):
                    self.Ds2[i].apply(weights_init(self.opt.init_type))
            if self.E_image is not None:
                self.E_image.apply(weights_init(self.opt.init_type))
                
            ################### use GPU #############################################
            self.G.cuda()
            for i in range(self.D_len):
                self.Ds[i].cuda()
            if self.Ds2 is not None:
                for i in range(self.D_len):
                    self.Ds2[i].cuda()
            if self.E_image is not None:
                self.E_image.cuda()
            if self.E_text is not None:
                self.E_text.cuda()
                
            ################### set criterion ########################################
            self.criterionGAN = GANLoss(mse_loss=True)
            if use_con:
                self.criterionCGAN = GANLoss(mse_loss = not use_sigmoid)
            self.criterionKL = KL_loss
            
            ################## define optimizers #####################################
            self.define_optimizers()
        
        
    def denorm(self,x):
        x = (x+1)/2
        return x.clamp_(0, 1)
        
    def define_optimizer(self, Net):
        return optim.Adam(Net.parameters(),
                                    lr=self.opt.lr,
                                    betas=(0.5, 0.999))
    def define_optimizers(self):
        self.G_opt = self.define_optimizer(self.G)
        self.Ds_opt = []
        self.Ds2_opt = []
        self.E_opt = None
        for i in range(self.D_len):
            self.Ds_opt.append(self.define_optimizer(self.Ds[i]))
        if self.Ds2 is not None:
            for i in range(self.D_len):
                self.Ds2_opt.append(self.define_optimizer(self.Ds2[i]))
        if self.E_image is not None:
            self.E_opt = self.define_optimizer(self.E_image)
    
    def update_lr(self, lr):
        for param_group in self.G_opt.param_groups:
            param_group['lr'] = lr
        for i in range(self.D_len):
            for param_group in self.Ds_opt[i].param_groups:
                param_group['lr'] = lr
        if self.E_opt is not None:
            for param_group in self.E_opt.param_groups:
                param_group['lr'] = lr
                
    def save(self, name):
        torch.save(self.G.state_dict(), '{}/G_{}.pth'.format(self.opt.model_dir, name))
        for i in range(self.D_len):
            torch.save(self.Ds[i].state_dict(), '{}/D_{}_{}.pth'.format(self.opt.model_dir, i, name))
        if self.E_image is not None:
            torch.save(self.E_image.state_dict(), '{}/E_image_{}.pth'.format(self.opt.model_dir, name))
            
    def get_c_random(self, size):
        c = torch.cuda.FloatTensor(size).normal_()
        return Variable(c)
    

    def prepare_label(self,data):
        imgs, c_global, c_local = data
        c_global = Variable(c_global).cuda()
        c_local = Variable(c_local).cuda()
        imgs_cuda = []
        for img in imgs:
            imgs_cuda.append(Variable(img).cuda())
        return [imgs_cuda, c_global, c_local]
        
    def prepare_image(self,data):
        imgAs, imgBs = data
        if isinstance(imgAs,list):
            imgA = []
            for img in imgAs:
                imgA.append(Variable(img).cuda())
        else:
            imgA = Variable(imgAs).cuda()
        imgB = []
        for img in imgBs:
            imgB.append(Variable(img).cuda())
        return [imgA, imgB]
    
    def prepare_text(self,data):
        imgs, captions, captions_lens, class_ids, keys = data
        sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)
        real_imgs = []
        for i in range(len(imgs)):
            imgs[i] = imgs[i][sorted_cap_indices]
            real_imgs.append(Variable(imgs[i]).cuda())
        captions = captions[sorted_cap_indices].squeeze()
        class_ids = class_ids[sorted_cap_indices].numpy()
        keys = [keys[i] for i in sorted_cap_indices.numpy()]
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()

        return [real_imgs, captions, sorted_cap_lens,
                    class_ids, keys]
            
    
    def get_current_errors(self):
        pass
    def get_current_visuals(self):
        pass
    def update_model(self,data):
        pass
