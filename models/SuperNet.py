from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
from .base_model import RAGAN
from collections import OrderedDict
import util.util as util
from torchvision.utils import save_image

# ################# Text to image task############################ #
class SuperNet(RAGAN):
    def name(self):
        return 'SupervisedResidualAttentionGAN'

    def initialize(self, opt):
        RAGAN.initialize(self,opt)
        if self.opt.c_type == 'image':
            self.update_model = self.update_model_image
        elif self.opt.c_type == 'text':
            assert self.opt.batchSize >= 2
            self.update_model = self.update_model_text
        elif self.opt.c_type == 'image_text':
            assert self.opt.batchSize >= 2
            self.update_model = self.update_model_image_text
        elif self.opt.c_type == 'label':
            assert self.opt.batchSize >= 2
            self.update_model = self.update_model_label
        elif self.opt.c_type == 'image_label':
            assert self.opt.batchSize >= 2
            self.update_model = self.update_model_image_label
        
    def get_current_errors(self):
        ret_dict = OrderedDict([('errD_total',self.errD_total.data[0]),
                                ('errG_total',self.errG_total.data[0])])
        for i in range(0,self.D_len):
            ret_dict['D_{}'.format(i)] = self.errD[i].data[0]
        for i in range(0,self.D_len):
            ret_dict['G_{}'.format(i)] = self.errG[i].data[0]
        ret_dict['errRec'] = self.errRec.data[0]
        if self.opt.c_type == 'image' or  self.opt.c_type == 'image_text' or self.opt.c_type == 'image_label':
            ret_dict['errKL'] = self.errKL.data[0]
            ret_dict['errCode'] = self.errCode.data[0]
            
        return ret_dict
            
    def get_current_visuals(self):
        if self.imgA.size(1)==1:
            self.imgA = self.imgA.repeat(1,3,1,1)
            
        fake_enc_0 = nn.Upsample(size=(self.opt.fineSize, self.opt.fineSize), mode='bilinear')(self.fake_enc[0])
        fake_enc_1 = nn.Upsample(size=(self.opt.fineSize, self.opt.fineSize), mode='bilinear')(self.fake_enc[1])

        imgA = util.tensor2im(self.imgA.data)
        imgB = util.tensor2im(self.imgB[-1].data)
        enc0 = util.tensor2im(fake_enc_0.data)
        enc1 = util.tensor2im(fake_enc_1.data)
        enc2 = util.tensor2im(self.fake_enc[2].data)
        ran  = util.tensor2im(self.fake_ran[-1].data)

        ret_dict = OrderedDict([('imgA', imgA),  ('imgB', imgB), ('enc0', enc0),
                                    ('enc1', enc1), ('enc2', enc2), ('ran', ran)])
        return ret_dict
        
    def sample_attnMap(self,name):
        map1, map2 = self.G.getAttnMap()
        if self.imgA.size(1)==1:
            self.imgA = self.imgA.repeat(1,3,1,1)
        imgA = nn.Upsample(size=(self.opt.fineSize//4, self.opt.fineSize//4), mode='bilinear')(self.imgA)
        enc1 = nn.Upsample(size=(self.opt.fineSize//4, self.opt.fineSize//4), mode='bilinear')(self.fake_enc[1])
        enc2 = nn.Upsample(size=(self.opt.fineSize//2, self.opt.fineSize//2), mode='bilinear')(self.fake_enc[2])
        imgB =  nn.Upsample(size=(self.opt.fineSize//2, self.opt.fineSize//2), mode='bilinear')(self.imgB[-1])
        map1 = torch.cat([imgA,self.fake_enc[0],map1,enc1],dim=3).data.cpu()
        map2 = torch.cat([self.fake_enc[1],map2,enc2,imgB],dim=3).data.cpu()
        
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
        fake_enc = self.G(imgA,c_glob,c_local,mask)
        fake_ran = self.G(imgA,c_glob.index_select(0,rand_idx),c_local.index_select(0,rand_idx),mask.index_select(0,rand_idx))
        ### train D ###
        self.errD = [.0,.0,.0]
        self.errD_total = 0.0
        for i in range(0,self.D_len):
            self.Ds[i].zero_grad()
            pred_fake1, c_fake1, _ = self.Ds[i](fake_enc[i].detach(),c=c_glob)
            pred_fake2, c_fake2, _ = self.Ds[i](fake_ran[i].detach(),c=c_glob.index_select(0,rand_idx))
            pred_real, c_real, x_code = self.Ds[i](imgB[i],c=c_glob)
            c_fake3 = self.Ds[i].c_pre(x_code, c_glob.index_select(0, rand_idx))
            self.errD[i] = (self.criterionGAN(pred_fake1,False)+self.criterionGAN(pred_fake2,False)+self.criterionGAN(pred_real,True))/3. + \
                           (self.criterionCGAN(c_fake1,False)+self.criterionCGAN(c_fake2,False)+self.criterionCGAN(c_fake3,False)+self.criterionCGAN(c_real,True))/4.
            self.errD[i].backward()
            self.Ds_opt[i].step()
            self.errD_total += self.errD[i]
        ### train G ###
        self.G.zero_grad()
        self.errG = [0.0,0.0,0.0]
        self.errG_total = 0.0
        self.errRec = 0.0
        recW = [0.33,0.67,1]
        for i in range(0,self.D_len):
            pred_fake1, c_fake1, _ = self.Ds[i](fake_enc[i],c=c_glob)
            pred_fake2, c_fake2, _ = self.Ds[i](fake_ran[i],c=c_glob.index_select(0,rand_idx))
            self.errG[i] = (self.criterionGAN(pred_fake1,True)+self.criterionGAN(pred_fake2,True))/2. + \
                           (self.criterionCGAN(c_fake1,True)+self.criterionCGAN(c_fake2,True))/2.
            self.errG_total += self.errG[i]
            self.errRec +=torch.mean(torch.abs(fake_enc[i]-imgB[i])) * self.opt.lambda_L1 *recW[i]
        self.errG_total += self.errRec
        self.errG_total.backward()
        self.G_opt.step()
        self.imgA = imgA
        self.imgB = imgB
        self.fake_enc = fake_enc
        self.fake_ran = fake_ran
    
    def update_model_label(self,data):
        imgs, c_glob, c_local = self.prepare_label(data)
        c_local = torch.transpose(c_local,1,2).contiguous()
        imgA = imgs[-1]
        imgB = imgs[0:3]
        rand_idx = Variable(torch.LongTensor([(n+1)%self.opt.batchSize for n in range(self.opt.batchSize) ])).cuda()
        ### generate image ###
        fake_enc = self.G(imgA,c_glob,c_local)
        fake_ran = self.G(imgA,c_glob.index_select(0,rand_idx),c_local.index_select(0,rand_idx))
        ### train D ###
        self.errD = [.0,.0,.0]
        self.errD_total = 0.0
        for i in range(0,self.D_len):
            self.Ds[i].zero_grad()
            pred_fake1, c_fake1, _ = self.Ds[i](fake_enc[i].detach(),c=c_glob)
            pred_fake2, c_fake2, _ = self.Ds[i](fake_ran[i].detach(),c=c_glob.index_select(0,rand_idx))
            pred_real, c_real, x_code = self.Ds[i](imgB[i],c=c_glob)
            c_fake3 = self.Ds[i].c_pre(x_code, c_glob.index_select(0, rand_idx))
            self.errD[i] = (self.criterionGAN(pred_fake1,False)+self.criterionGAN(pred_fake2,False)+self.criterionGAN(pred_real,True))/3. + \
                           (self.criterionCGAN(c_fake1,False)+self.criterionCGAN(c_fake2,False)+self.criterionCGAN(c_fake3,False)+self.criterionCGAN(c_real,True))/4.
            self.errD[i].backward()
            self.Ds_opt[i].step()
            self.errD_total += self.errD[i]
        ### train G ###
        self.G.zero_grad()
        self.errG = [0.0,0.0,0.0]
        self.errG_total = 0.0
        self.errRec = 0.0
        recW = [0.33,0.67,1]
        for i in range(0,self.D_len):
            pred_fake1, c_fake1, _ = self.Ds[i](fake_enc[i],c=c_glob)
            pred_fake2, c_fake2, _ = self.Ds[i](fake_ran[i],c=c_glob.index_select(0,rand_idx))
            self.errG[i] = (self.criterionGAN(pred_fake1,True)+self.criterionGAN(pred_fake2,True))/2. + \
                           (self.criterionCGAN(c_fake1,True)+self.criterionCGAN(c_fake2,True))/2.
            self.errG_total += self.errG[i]
            self.errRec +=torch.mean(torch.abs(fake_enc[i]-imgB[i])) * self.opt.lambda_L1 *recW[i]
        self.errG_total += self.errRec
        self.errG_total.backward()
        self.G_opt.step()
        self.imgA = imgA
        self.imgB = imgB
        self.fake_enc = fake_enc
        self.fake_ran = fake_ran
    
    def update_model_image_label(self,data):
        imgs, c_glob, c_local = self.prepare_label(data)
        c_local = torch.transpose(c_local,1,2).contiguous()
        imgA = imgs[-1]
        imgB = imgs[0:3]
        rand_idx = Variable(torch.LongTensor([(n+1)%self.opt.batchSize for n in range(self.opt.batchSize) ])).cuda()
        ### generate image ###
        e_enc, mu, logvar = self.E_image(imgB[-1])
        e_rand = self.get_c_random(e_enc.size())
        fake_enc = self.G(imgA,c_glob,c_local,e_global=e_enc)
        fake_ran = self.G(imgA,c_glob.index_select(0,rand_idx),c_local.index_select(0,rand_idx),e_global=e_rand)
        _, mu_enc, _ = self.E_image(fake_ran[-1])
        ### train D ###
        self.errD = [.0,.0,.0]
        self.errD_total = 0.0
        for i in range(0,self.D_len):
            self.Ds[i].zero_grad()
            pred_fake1, c_fake1, _ = self.Ds[i](fake_enc[i].detach(),c=c_glob)
            pred_fake2, c_fake2, _ = self.Ds[i](fake_ran[i].detach(),c=c_glob.index_select(0,rand_idx))
            pred_real, c_real, x_code = self.Ds[i](imgB[i],c=c_glob)
            c_fake3 = self.Ds[i].c_pre(x_code, c_glob.index_select(0, rand_idx))
            self.errD[i] = (self.criterionGAN(pred_fake1,False)+self.criterionGAN(pred_fake2,False)+self.criterionGAN(pred_real,True))/3. + \
                           (self.criterionCGAN(c_fake1,False)+self.criterionCGAN(c_fake2,False)+self.criterionCGAN(c_fake3,False)+self.criterionCGAN(c_real,True))/4.
            self.errD[i].backward()
            self.Ds_opt[i].step()
            self.errD_total += self.errD[i]
        ### train G ###
        self.G.zero_grad()
        self.E_image.zero_grad()
        self.errG = [0.0,0.0,0.0]
        self.errG_total = 0.0
        self.errRec = 0.0
        recW = [0.33,0.67,1]
        for i in range(0,self.D_len):
            pred_fake1, c_fake1, _ = self.Ds[i](fake_enc[i],c=c_glob)
            pred_fake2, c_fake2, _ = self.Ds[i](fake_ran[i],c=c_glob.index_select(0,rand_idx))
            self.errG[i] = (self.criterionGAN(pred_fake1,True)+self.criterionGAN(pred_fake2,True))/2. + \
                           (self.criterionCGAN(c_fake1,True)+self.criterionCGAN(c_fake2,True))/2.
            self.errG_total += self.errG[i]
            self.errRec +=torch.mean(torch.abs(fake_enc[i]-imgB[i])) * self.opt.lambda_L1 * recW[i]
        self.errKL = self.criterionKL(mu,logvar) * self.opt.lambda_kl
        self.errG_total += self.errRec + self.errKL
        self.errG_total.backward(retain_graph=True)
        self.G_opt.step()
        self.E_opt.step()
        self.G.zero_grad()
        self.E_image.zero_grad()
        self.errCode = torch.mean(torch.abs(mu_enc - e_rand)) * self.opt.lambda_c
        self.errCode.backward()
        self.G_opt.step()
        self.errG_total += self.errCode
        self.imgA = imgA
        self.imgB = imgB
        self.fake_enc = fake_enc
        self.fake_ran = fake_ran
    
    def update_model_image_text(self,data):
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
        e_enc, mu, logvar = self.E_image(imgB[-1])
        e_rand = self.get_c_random(e_enc.size())
        fake_enc = self.G(imgA,c_glob,c_local,mask,e_global=e_enc)
        fake_ran = self.G(imgA,c_glob.index_select(0,rand_idx),c_local.index_select(0,rand_idx),mask.index_select(0,rand_idx),e_global=e_rand)
        _, mu_enc, _ = self.E_image(fake_ran[-1])
        ### train D ###
        self.errD = [.0,.0,.0]
        self.errD_total = 0.0
        for i in range(0,self.D_len):
            self.Ds[i].zero_grad()
            pred_fake1, c_fake1, _ = self.Ds[i](fake_enc[i].detach(),c=c_glob)
            pred_fake2, c_fake2, _ = self.Ds[i](fake_ran[i].detach(),c=c_glob.index_select(0,rand_idx))
            pred_real, c_real, x_code = self.Ds[i](imgB[i],c=c_glob)
            c_fake3 = self.Ds[i].c_pre(x_code, c_glob.index_select(0, rand_idx))
            self.errD[i] = (self.criterionGAN(pred_fake1,False)+self.criterionGAN(pred_fake2,False)+self.criterionGAN(pred_real,True))/3. + \
                           (self.criterionCGAN(c_fake1,False)+self.criterionCGAN(c_fake2,False)+self.criterionCGAN(c_fake3,False)+self.criterionCGAN(c_real,True))/4.
            self.errD[i].backward()
            self.Ds_opt[i].step()
            self.errD_total += self.errD[i]
        ### train G ###
        self.G.zero_grad()
        self.E_image.zero_grad()
        self.errG = [0.0,0.0,0.0]
        self.errG_total = 0.0
        self.errRec = 0.0
        recW = [0.33,0.67,1]
        for i in range(0,self.D_len):
            pred_fake1, c_fake1, _ = self.Ds[i](fake_enc[i],c=c_glob)
            pred_fake2, c_fake2, _ = self.Ds[i](fake_ran[i],c=c_glob.index_select(0,rand_idx))
            self.errG[i] = (self.criterionGAN(pred_fake1,True)+self.criterionGAN(pred_fake2,True))/2. + \
                           (self.criterionCGAN(c_fake1,True)+self.criterionCGAN(c_fake2,True))/2.
            self.errG_total += self.errG[i]
            self.errRec +=torch.mean(torch.abs(fake_enc[i]-imgB[i])) * self.opt.lambda_L1 * recW[i]
        self.errKL = self.criterionKL(mu,logvar) * self.opt.lambda_kl
        self.errG_total += self.errRec + self.errKL
        self.errG_total.backward(retain_graph=True)
        self.G_opt.step()
        self.E_opt.step()
        self.G.zero_grad()
        self.E_image.zero_grad()
        self.errCode = torch.mean(torch.abs(mu_enc - e_rand)) * self.opt.lambda_c
        self.errCode.backward()
        self.G_opt.step()
        self.errG_total += self.errCode
        self.imgA = imgA
        self.imgB = imgB
        self.fake_enc = fake_enc
        self.fake_ran = fake_ran
    
    def update_model_image(self,data):
        imgA, imgB = self.prepare_image(data)
        ### generate image ###
        c_glob, c_local, mu, logvar = self.E_image(imgB[-1])
        c_rand_local = self.get_c_random(c_local.size())
        c_rand_glob = torch.mean(c_rand_local,dim=2)
        fake_enc = self.G(imgA,c_glob,c_local)
        fake_ran = self.G(imgA,c_rand_glob,c_rand_local)
        c_rand_glob_enc, _, _, _ = self.E_image(fake_ran[-1])
        ### train D ###
        self.errD = [0.0,.0,.0]
        self.errD_total = 0.0
        for i in range(0,self.D_len):
            self.Ds[i].zero_grad()
            pred_fake1, _, _ = self.Ds[i](fake_enc[i].detach())
            pred_fake2, _, _ = self.Ds[i](fake_ran[i].detach())
            pred_real, _, _ = self.Ds[i](imgB[i])
            self.errD[i] = (self.criterionGAN(pred_fake1,False)+self.criterionGAN(pred_fake2,False)+self.criterionGAN(pred_real,True))/3.
            self.errD[i].backward()
            self.Ds_opt[i].step()
            self.errD_total += self.errD[i]
        ### train G ###
        self.G.zero_grad()
        self.E_image.zero_grad()
        self.errG = [0.0,0.0,0.0]
        self.errG_total = 0.0
        self.errRec = 0.0
        recW = [0.33,0.67,1]
        for i in range(0,self.D_len):
            pred_fake1, _, _ = self.Ds[i](fake_enc[i])
            pred_fake2, _, _ = self.Ds[i](fake_ran[i])
            self.errG[i] = (self.criterionGAN(pred_fake1,True)+self.criterionGAN(pred_fake2,True))/2.
            self.errG_total += self.errG[i]
            self.errRec +=torch.mean(torch.abs(fake_enc[i]-imgB[i])) * self.opt.lambda_L1 *recW[i]
        self.errKL = self.criterionKL(mu,logvar) * self.opt.lambda_kl
        self.errG_total += self.errRec + self.errKL
        self.errG_total.backward(retain_graph=True)
        self.G_opt.step()
        self.E_opt.step()
        self.G.zero_grad()
        self.E_image.zero_grad()
        self.errCode = torch.mean(torch.abs(c_rand_glob_enc - c_rand_glob)) * self.opt.lambda_c
        self.errCode.backward()
        self.G_opt.step()
        self.errG_total += self.errCode
        self.imgA = imgA
        self.imgB = imgB
        self.fake_enc = fake_enc
        self.fake_ran = fake_ran
        
        
        
