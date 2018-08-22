from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
from .base_model import RAGAN
from collections import OrderedDict
import util.util as util
from torchvision.utils import save_image

# ################# Text to image task############################ #
class USuperNet(RAGAN):
    def name(self):
        return 'UnSupervisedResidualAttentionGAN'

    def initialize(self, opt):
        RAGAN.initialize(self,opt)
        if self.opt.c_type == 'image':
            self.update_model = self.update_model_image
        elif self.opt.c_type == 'text':
            assert self.opt.batchSize >= 2
            self.update_model = self.update_model_text
        elif self.opt.c_type == 'label':
            self.update_model = self.update_model_label
        
    def get_current_errors(self):
        ret_dict = OrderedDict([('errD_total',self.errD_total.data[0]),
                                ('errG_total',self.errG_total.data[0])])
        for i in range(0,self.D_len):
            ret_dict['D_{}'.format(i)] = self.errD[i].data[0]
        for i in range(0,self.D_len):
            ret_dict['G_{}'.format(i)] = self.errG[i].data[0]
        ret_dict['errRec'] = self.errRec.data[0]
        if self.opt.c_type == 'image':
            ret_dict['errKL'] = self.errKL.data[0]
            
        return ret_dict
            
    def get_current_visuals(self):

        fake_ran_0 = nn.Upsample(size=(self.opt.fineSize, self.opt.fineSize), mode='bilinear')(self.fake_ran[0])
        fake_ran_1 = nn.Upsample(size=(self.opt.fineSize, self.opt.fineSize), mode='bilinear')(self.fake_ran[1])

        imgA = util.tensor2im(self.imgA.data)
        fake0 = util.tensor2im(fake_ran_0.data)
        fake1 = util.tensor2im(fake_ran_1.data)
        fake2 = util.tensor2im(self.fake_ran[2].data)
        rec = util.tensor2im(self.fake_rec[2].data)
        imgB = util.tensor2im(self.imgB.data)
        dict = [('imgA', imgA),  ('fake0', fake0), ('fake1', fake1),
                                    ('fake2', fake2), ('rec', rec), ('imgB', imgB)]
        if self.opt.c_type == 'image':
            fake_A = util.tensor2im(self.fake_ran[-1][self.opt.batchSize:].data)
            dict.append(('fake_A',fake_A))

        ret_dict = OrderedDict(dict)
        return ret_dict
        
    def sample_attnMap(self,name):
        map1, map2 = self.G.getAttnMap()
        if self.imgA.size(1)==1:
            self.imgA = self.imgA.repeat(1,3,1,1)
        imgA = nn.Upsample(size=(self.opt.fineSize//4, self.opt.fineSize//4), mode='bilinear')(self.imgA)
        fake1 = nn.Upsample(size=(self.opt.fineSize//4, self.opt.fineSize//4), mode='bilinear')(self.fake_ran[1])
        fake2 = nn.Upsample(size=(self.opt.fineSize//2, self.opt.fineSize//2), mode='bilinear')(self.fake_ran[2])
        imgB =  nn.Upsample(size=(self.opt.fineSize//2, self.opt.fineSize//2), mode='bilinear')(self.imgB)
        map1 = torch.cat([imgA,self.fake_ran[0],map1,fake1],dim=3).data.cpu()
        map2 = torch.cat([self.fake_ran[1],map2,fake2,imgB],dim=3).data.cpu()
        
        save_image(self.denorm(map1),self.opt.sample_dir+'/map{}_1.jpg'.format(name), nrow=1, padding=0)
        save_image(self.denorm(map2),self.opt.sample_dir+'/map{}_2.jpg'.format(name), nrow=1, padding=0)
        
    
        
    def update_model_text(self,data):
        imgs, captions, cap_lens, _, _ = self.prepare_text(data)
        imgA = imgs[-1]
        imgB = imgs[0:3]
        rand_idx = Variable(torch.LongTensor([(n+1)%self.opt.batchSize for n in range(self.opt.batchSize) ])).cuda()
        hidden = self.E_text.init_hidden(self.opt.batchSize)
        words_embs, sent_emb = self.E_text(captions, cap_lens, hidden)
        c_glob = sent_emb.detach()
        c_local = words_embs.detach()
        mask = (captions == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]
        ### generate image ###
        fake_ran = self.G(imgA,c_glob.index_select(0,rand_idx),c_local.index_select(0,rand_idx),mask.index_select(0,rand_idx))
        fake_rec = self.G(fake_ran[-1],c_glob,c_local,mask)
        fake_ide = self.G(imgA,c_glob,c_local,mask)
        ### train D ###
        self.errD = [.0,.0,.0]
        self.errD_total = 0.0
        for i in range(0,self.D_len):
            self.Ds[i].zero_grad()
            pred_fake1, c_fake1, _ = self.Ds[i](fake_ran[i].detach(),c=c_glob.index_select(0,rand_idx))
            pred_real, c_real, x_code = self.Ds[i](imgB[i],c=c_glob)
            c_fake2 = self.Ds[i].c_pre(x_code, c_glob.index_select(0, rand_idx))
            self.errD[i] = (self.criterionGAN(pred_fake1,False)+self.criterionGAN(pred_real,True))/2. + \
                           (self.criterionCGAN(c_fake1,False)+self.criterionCGAN(c_fake2,False)+self.criterionCGAN(c_real,True))/3.
            self.errD[i].backward()
            self.Ds_opt[i].step()
            self.errD_total += self.errD[i]
        ### train G ###
        self.G.zero_grad()
        self.errG = [0.0,0.0,0.0]
        self.errG_total, self.errRec = 0.0, 0.0
	recW = [0.33,0.67,1]
        for i in range(0,self.D_len):
            pred_fake, c_fake, _ = self.Ds[i](fake_ran[i],c=c_glob.index_select(0,rand_idx))
            self.errG[i] = self.criterionGAN(pred_fake,True)+self.criterionCGAN(c_fake,True)
            self.errG_total += self.errG[i]
            self.errRec += torch.mean(torch.abs(imgB[i] - fake_rec[i])) * self.opt.lambda_L1 *recW[i]
        self.errRec += torch.mean(torch.abs(imgB[-1] - fake_ide[-1])) * self.opt.lambda_L1
        self.errG_total += self.errRec
        self.errG_total.backward()
        self.G_opt.step()
        self.imgA = imgA
        self.imgB = imgB[-1].index_select(0,rand_idx)
        self.fake_rec = fake_rec
        self.fake_ran = fake_ran
    
    def update_model_label(self,data):
        imgs, c_glob, c_local = self.prepare_label(data)
        c_local = torch.transpose(c_local,1,2).contiguous()
        imgA = imgs[-1]
        imgB = imgs[0:3]
        rand_idx = Variable(torch.LongTensor([(n+1)%self.opt.batchSize for n in range(self.opt.batchSize) ])).cuda()
        ### generate image ###
        fake_ran = self.G(imgA,c_glob.index_select(0,rand_idx),c_local.index_select(0,rand_idx))
        fake_rec = self.G(fake_ran[-1],c_glob,c_local)
        ### train D ###
        self.errD = [.0,.0,.0]
        self.errD_total = 0.0
        for i in range(0,self.D_len):
            self.Ds[i].zero_grad()
            pred_fake1, c_fake1, _ = self.Ds[i](fake_ran[i].detach(),c=c_glob.index_select(0,rand_idx))
            pred_real, c_real, x_code = self.Ds[i](imgB[i],c=c_glob)
            c_fake2 = self.Ds[i].c_pre(x_code, c_glob.index_select(0, rand_idx))
            self.errD[i] = (self.criterionGAN(pred_fake1,False)+self.criterionGAN(pred_real,True))/2. + \
                           (self.criterionCGAN(c_fake1,False)+self.criterionCGAN(c_fake2,False)+self.criterionCGAN(c_real,True))/3.
            self.errD[i].backward()
            self.Ds_opt[i].step()
            self.errD_total += self.errD[i]
        ### train G ###
        self.G.zero_grad()
        self.errG = [0.0,0.0,0.0]
        self.errG_total, self.errRec = 0.0, 0.0
	recW = [0.33,0.67,1]
        for i in range(0,self.D_len):
            pred_fake, c_fake, _ = self.Ds[i](fake_ran[i],c=c_glob.index_select(0,rand_idx))
            self.errG[i] = self.criterionGAN(pred_fake,True)+self.criterionCGAN(c_fake,True)
            self.errG_total += self.errG[i]
            self.errRec += torch.mean(torch.abs(imgB[i] - fake_rec[i])) * self.opt.lambda_L1 * recW[i]
        self.errG_total += self.errRec
        self.errG_total.backward()
        self.G_opt.step()
        self.imgA = imgA
        self.imgB = imgB[-1].index_select(0,rand_idx)
        self.fake_rec = fake_rec
        self.fake_ran = fake_ran
        
    
    def update_model_image(self,data):
        img1, img2 = self.prepare_image(data)
        e_code = self.get_domain_code(self.opt.batchSize,2)
        e_code_cyc = 1. - e_code
        img = []
        for index in range(3):
            img.append(torch.cat([img1[index],img2[index]],0))
        ### generate image ###
        c_glob, c_local, mu, logvar = self.E_image(img[-1], e_code_cyc)
        c_rand_local = self.get_c_random(c_local.size())
        c_rand_glob = torch.mean(c_rand_local,dim=2)
        fake = self.G(img[-1],c_rand_glob,c_rand_local,e_global=e_code)
        rec = self.G(fake[-1],c_glob,c_local,e_global=e_code_cyc)
        c_glob_rand, _, _, _ = self.E_image(fake[-1], e_code)
        
        ### train D ###
        self.errD, self.errD_total = [.0,.0,.0], .0
        for i in range(0,self.D_len):
            self.Ds[i].zero_grad()
            self.Ds2[i].zero_grad()
            pred_fake, _, _ = self.Ds[i](fake[i][:self.opt.batchSize].detach())
            pred_real, _, _ = self.Ds[i](img[i][self.opt.batchSize:])
            self.errD[i] += self.criterionGAN(pred_fake,False)+self.criterionGAN(pred_real,True)
            pred_fake, _, _ = self.Ds2[i](fake[i][self.opt.batchSize:].detach())
            pred_real, _, _ = self.Ds2[i](img[i][:self.opt.batchSize])
            self.errD[i] += self.criterionGAN(pred_fake,False)+self.criterionGAN(pred_real,True)
            self.errD[i].backward()
            self.Ds_opt[i].step()
            self.Ds2_opt[i].step()
            self.errD_total += self.errD[i]
  
        ### train G ###
        self.G.zero_grad()
        self.E_image.zero_grad()
        self.errG, self.errG_total, self.errRec = [.0,.0,.0], .0, .0
	recW = [0.33,0.67,1]
        for i in range(0,self.D_len):
            pred_fake, _, _ = self.Ds[i](fake[i][:self.opt.batchSize])
            pred_fake2, _, _ = self.Ds2[i](fake[i][self.opt.batchSize:])
            self.errG[i] = self.criterionGAN(pred_fake,True)+self.criterionGAN(pred_fake2,True) 
            self.errG_total += self.errG[i]
            self.errRec += torch.mean(torch.abs(rec[i]-img[i])) *  self.opt.lambda_L1 * recW[i]
            
        self.errKL = self.criterionKL(mu,logvar) * self.opt.lambda_kl
        self.errG_total += self.errKL + self.errRec
        
        self.errG_total.backward(retain_graph=True)
        self.G_opt.step()
        self.E_opt.step()
        self.G.zero_grad()
        self.E_image.zero_grad()
        self.errCode = torch.mean(torch.abs(c_glob_rand - c_rand_glob)) * self.opt.lambda_c
        self.errCode.backward()
        self.G_opt.step()
        self.errG_total += self.errCode
        self.imgA = img[-1]
        self.imgB = torch.cat([img2[-1],img1[-1]],dim=0)
        self.fake_ran = fake
        self.fake_rec = rec
        
    def get_domain_code(self,size,category):
        codes = []
        for index in range(category):
            c = torch.zeros([size,category])
            c[:,index] = 1
            codes.append(c)
        codes = torch.cat(codes,dim=0)
        return Variable(codes).cuda()
        
        
        
