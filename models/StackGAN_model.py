#==================================
# MC-GAN
# End-to-End GlyphNet + OrnaNet
# By Samaneh Azadi
#==================================

import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from scipy import misc
from torch import index_select, LongTensor
import random
import torchvision.transforms as transforms




class StackGANModel(BaseModel):
    def name(self):
        return 'StackGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # define tensors
        self.input_A0 = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B0 = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        self.input_base = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)


        # load/define networks
        if self.opt.conv3d:
            # one layer for considering a conv filter for each of the 26 channels
            self.netG_3d = networks.define_G_3d(opt.input_nc, opt.input_nc, norm=opt.norm, groups=opt.grps, gpu_ids=self.gpu_ids)

        # Generator of the GlyphNet
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)

        
        #Generator of the OrnaNet as an Encoder and a Decoder
        self.netE1 = networks.define_Enc(opt.input_nc_1, opt.output_nc_1, opt.ngf,
                                    opt.which_model_netG, opt.norm, opt.use_dropout1, self.gpu_ids)
        
        self.netDE1 = networks.define_Dec(opt.input_nc_1, opt.output_nc_1, opt.ngf,
                                    opt.which_model_netG, opt.norm, opt.use_dropout1, self.gpu_ids)
                            

        if self.opt.conditional:
            # not applicable for non-conditional case
            use_sigmoid = opt.no_lsgan
            if opt.which_model_preNet != 'none':
                self.preNet_A = networks.define_preNet(self.opt.input_nc_1+self.opt.output_nc_1, self.opt.input_nc_1+self.opt.output_nc_1, which_model_preNet=opt.which_model_preNet,norm=opt.norm, gpu_ids=self.gpu_ids)

            nif = opt.input_nc_1+opt.output_nc_1

            
            netD_norm = opt.norm

            self.netD1 = networks.define_D(nif, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, netD_norm, use_sigmoid, True, self.gpu_ids)



        if self.isTrain:
            if self.opt.conv3d:
                 self.load_network(self.netG_3d, 'G_3d', opt.which_epoch)

            self.load_network(self.netG, 'G', opt.which_epoch)

            if self.opt.print_weights:
                for key in self.netE1.state_dict().keys():
                    print key, 'random_init, mean,std:', torch.mean(self.netE1.state_dict()[key]),torch.std(self.netE1.state_dict()[key])
                for key in self.netDE1.state_dict().keys():
                    print key, 'random_init, mean,std:', torch.mean(self.netDE1.state_dict()[key]),torch.std(self.netDE1.state_dict()[key])


        if not self.isTrain:
            print "Load generators from their pretrained models..."
            if opt.no_Style2Glyph:
                if self.opt.conv3d:
                     self.load_network(self.netG_3d, 'G_3d', opt.which_epoch)
                self.load_network(self.netG, 'G', opt.which_epoch)
                self.load_network(self.netE1, 'E1', opt.which_epoch1)
                self.load_network(self.netDE1, 'DE1', opt.which_epoch1)
                self.load_network(self.netD1, 'D1', opt.which_epoch1)
                if opt.which_model_preNet != 'none':
                    self.load_network(self.preNet_A, 'PRE_A', opt.which_epoch1)
            else:
                if self.opt.conv3d:
                     self.load_network(self.netG_3d, 'G_3d', str(int(opt.which_epoch)+int(opt.which_epoch1)))
                self.load_network(self.netG, 'G', str(int(opt.which_epoch)+int(opt.which_epoch1)))
                self.load_network(self.netE1, 'E1', str(int(opt.which_epoch1)))
                self.load_network(self.netDE1, 'DE1', str(int(opt.which_epoch1)))
                self.load_network(self.netD1, 'D1', str(int(opt.which_epoch1)))
                if opt.which_model_preNet != 'none':
                    self.load_network(self.preNet_A, 'PRE_A', opt.which_epoch1)


        if self.isTrain:
            if opt.continue_train:
                print "Load StyleNet from its pretrained model..."
                self.load_network(self.netE1, 'E1', opt.which_epoch1)
                self.load_network(self.netDE1, 'DE1', opt.which_epoch1)
                self.load_network(self.netD1, 'D1', opt.which_epoch1)
                if opt.which_model_preNet != 'none':
                    self.load_network(self.preNet_A, 'PRE_A', opt.which_epoch1)


        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        if self.isTrain:
            self.fake_AB1_pool = ImagePool(opt.pool_size)

            self.old_lr = opt.lr
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()


            # initialize optimizers
            if self.opt.conv3d:
                 self.optimizer_G_3d = torch.optim.Adam(self.netG_3d.parameters(),
                                                     lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_E1 = torch.optim.Adam(self.netE1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.which_model_preNet != 'none':
                self.optimizer_preA = torch.optim.Adam(self.preNet_A.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))

                                            
            self.optimizer_DE1 = torch.optim.Adam(self.netDE1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))


            print('---------- Networks initialized -------------')
            if self.opt.conv3d:
                networks.print_network(self.netG_3d)
            networks.print_network(self.netG)
            networks.print_network(self.netE1)
            networks.print_network(self.netDE1)
            if opt.which_model_preNet != 'none':
                networks.print_network(self.preNet_A)

            networks.print_network(self.netD1)
            print('-----------------------------------------------')

            self.initial = True

    def set_input(self, input):
        input_A0 = input['A']
        input_B0 = input['B']
        input_base = input['A_base']
        
        self.input_A0.resize_(input_A0.size()).copy_(input_A0)
        self.input_B0.resize_(input_B0.size()).copy_(input_B0)
        self.input_base.resize_(input_base.size()).copy_(input_base)
        self.image_paths = input['B_paths']


        b,c,m,n = self.input_base.size()
        
        real_base = self.Tensor(self.opt.output_nc,self.opt.input_nc_1, m,n)
        for batch in range(self.opt.output_nc):
            if not self.opt.rgb_in and self.opt.rgb_out:
                real_base[batch,0,:,:] = self.input_base[0,batch,:,:]
                real_base[batch,1,:,:] = self.input_base[0,batch,:,:]
                real_base[batch,2,:,:] = self.input_base[0,batch,:,:]
        
        self.real_base = Variable(real_base, requires_grad=False)

        if self.opt.isTrain:

            self.id_ = {}
            self.obs = []
            for i,im in enumerate(self.image_paths):
                self.id_[int(im.split('/')[-1].split('.png')[0].split('_')[-1])]=i
                self.obs += [int(im.split('/')[-1].split('.png')[0].split('_')[-1])]
            for i in list(set(range(self.opt.output_nc))-set(self.obs)):
                self.id_[i] = np.random.randint(low=0, high=len(self.image_paths))

            self.num_disc = self.opt.output_nc +1


    def all2observed(self, tensor_all):
        b,c,m,n = self.real_A0.size()

        self.out_id = self.obs
        tensor_gt = self.Tensor(b,self.opt.input_nc_1, m,n)
        for batch in range(b):
            if not self.opt.rgb_in and self.opt.rgb_out:
                tensor_gt[batch,0,:,:] = tensor_all.data[batch,self.out_id[batch],:,:]
                tensor_gt[batch,1,:,:] = tensor_all.data[batch,self.out_id[batch],:,:]
                tensor_gt[batch,2,:,:] = tensor_all.data[batch,self.out_id[batch],:,:]
            else:
                #TODO
                tensor_gt[batch,:,:,:] = tensor_all.data[batch,self.out_id[batch]*np.array(self.opt.input_nc_1):(self.out_id[batch]+1)*np.array(self.opt.input_nc_1),:,:]
        return tensor_gt

    def forward0(self):
        self.real_A0 = Variable(self.input_A0)
        if self.opt.conv3d:
            self.real_A0_indep = self.netG_3d.forward(self.real_A0.unsqueeze(2))
            self.fake_B0 = self.netG.forward(self.real_A0_indep.squeeze(2))
        else:
            self.fake_B0 = self.netG.forward(self.real_A0)
        if self.initial:
            if self.opt.orna:
                self.fake_B0_init = self.real_A0
            else:
                self.fake_B0_init = self.fake_B0



                                
    def forward1(self, inp_grad=False):
        b,c,m,n = self.real_A0.size()
        
        self.batch_ = b
        self.out_id = self.obs
        real_A1 = self.Tensor(self.opt.output_nc,self.opt.input_nc_1, m,n)
        if self.opt.orna:
            inp_orna = self.fake_B0_init
        else:
            inp_orna = self.fake_B0

        for batch in range(self.opt.output_nc):
            if not self.opt.rgb_in and self.opt.rgb_out:
                real_A1[batch,0,:,:] = inp_orna.data[self.id_[batch],batch,:,:]
                real_A1[batch,1,:,:] = inp_orna.data[self.id_[batch],batch,:,:]
                real_A1[batch,2,:,:] = inp_orna.data[self.id_[batch],batch,:,:]
            else:
                #TODO
                real_A1[batch,:,:,:] = inp_orna.data[batch,self.out_id[batch]*np.array(self.opt.input_nc_1):(self.out_id[batch]+1)*np.array(self.opt.input_nc_1),:,:]
        if self.initial:
            self.real_A1_init = Variable(real_A1, requires_grad=False)
            self.initial = False

        self.real_A1_s = Variable(real_A1, requires_grad=inp_grad)
        self.real_A1 = self.real_A1_s

        self.fake_B1_emb = self.netE1.forward(self.real_A1)
        self.fake_B1 = self.netDE1.forward(self.fake_B1_emb)
        self.real_B1 = Variable(self.input_B0)

        self.real_A1_gt_s = Variable(self.all2observed(inp_orna), requires_grad=True)
        self.real_A1_gt = (self.real_A1_gt_s)

        self.fake_B1_gt_emb = self.netE1.forward(self.real_A1_gt)
        self.fake_B1_gt = self.netDE1.forward(self.fake_B1_gt_emb)

        obs_ = torch.cuda.LongTensor(self.obs) if self.opt.gpu_ids else LongTensor(self.obs)

        real_base_gt = index_select(self.real_base, 0, obs_)
        self.real_base_gt = (Variable(real_base_gt.data, requires_grad=False))


    def add_noise_disc(self,real):
        #add noise to the discriminator target labels
        #real: True/False? 
        if self.opt.noisy_disc:
            rand_lbl = random.random()
            if rand_lbl<0.6:
                label = (not real)
            else:
                label = (real)
        else:  
            label = (real)
        return label
            
                
    
    # no backprop gradients
    def test(self):
        self.real_A0 = Variable(self.input_A0, volatile=True)

        if self.opt.conv3d:
            self.real_A0_indep = self.netG_3d.forward(self.real_A0.unsqueeze(2))
            self.fake_B0 = self.netG.forward(self.real_A0_indep.squeeze(2))
        else:
            self.fake_B0 = self.netG.forward(self.real_A0)            

        b,c,m,n = self.fake_B0.size()
        
        #for test time: we need to generate output for all of the glyphs in each input image
        if self.opt.rgb_in:
            self.batch_ = c/self.opt.input_nc_1
        else:
            self.batch_ = c
        self.out_id = range(self.batch_)
        real_A1 = self.Tensor(self.batch_,self.opt.input_nc_1, m,n)

        
        if self.opt.orna:
            inp_orna = self.real_A0
        else:
            inp_orna = self.fake_B0 
        for batch in range(self.batch_):
            if not self.opt.rgb_in and self.opt.rgb_out:
                real_A1[batch,0,:,:] = inp_orna.data[:,self.out_id[batch],:,:]
                real_A1[batch,1,:,:] = inp_orna.data[:,self.out_id[batch],:,:]
                real_A1[batch,2,:,:] = inp_orna.data[:,self.out_id[batch],:,:]
            else:
                real_A1[batch,:,:,:] = inp_orna.data[:,self.out_id[batch]*np.array(self.opt.input_nc_1):(self.out_id[batch]+1)*np.array(self.opt.input_nc_1),:,:]



        self.real_A1 = Variable(real_A1, volatile=True)
    
        fake_B1_emb = self.netE1.forward(self.real_A1.detach())
        self.fake_B1 = self.netDE1.forward(fake_B1_emb)
        
        self.real_B1 = Variable(self.input_B0, volatile=True)


    #get image paths
    def get_image_paths(self):
        return self.image_paths


    def prepare_data(self):
        if self.opt.conditional:
            if self.opt.base_font:
                self.first_pair = self.real_base
                self.first_pair_gt = self.real_base_gt
            else:
                self.first_pair = Variable(self.real_A1.data, requires_grad=False)
                self.first_pair_gt = Variable(self.real_A1_gt.data,requires_grad=False)


    def backward_D1(self):
        b,c,m,n = self.fake_B1.size()
    
        # Fake
        # stop backprop to the generator by detaching fake_B
        label_fake = self.add_noise_disc(False)
        if self.opt.conditional:

            fake_AB1 = self.fake_AB1_pool.query(torch.cat((self.first_pair, self.fake_B1),1))
            self.pred_fake1 = self.netD1.forward(fake_AB1.detach())
            if self.opt.which_model_preNet != 'none':
                #transform the input
                transformed_AB1 = self.preNet_A.forward(fake_AB1.detach())
                self.pred_fake_GL = self.netD1.forward(transformed_AB1)

            self.loss_D1_fake = 0
            self.loss_D1_fake += self.criterionGAN(self.pred_fake1, label_fake) 
            
            if self.opt.which_model_preNet != 'none':
                self.loss_D1_fake += self.criterionGAN(self.pred_fake_GL, label_fake)            
           


        # Real
        label_real = self.add_noise_disc(True)
        if self.opt.conditional:

            real_AB1 = torch.cat((self.first_pair_gt, self.real_B1), 1).detach()                
            self.pred_real1 = self.netD1.forward(real_AB1)

            if self.opt.which_model_preNet != 'none':
                transformed_real_AB1 = self.preNet_A.forward(real_AB1)
                self.pred_real1_GL = self.netD1.forward(transformed_real_AB1)


            self.loss_D1_real = 0
            self.loss_D1_real += self.criterionGAN(self.pred_real1, label_real)    
            if self.opt.which_model_preNet != 'none':                    
                self.loss_D1_real += self.criterionGAN(self.pred_real1_GL, label_real)    

        
        # Combined loss
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * 0.5
        self.loss_D1.backward()


    def backward_G(self, pass_grad, iter):

        b,c,m,n = self.fake_B0.size()
        if not self.opt.lambda_C or (iter>700):
            self.loss_G_L1 = Variable(torch.zeros(1))

        else:
            weight_val = 10.0

            weights = torch.ones(b,c,m,n).cuda() if self.opt.gpu_ids else torch.ones(b,c,m,n)
            obs_ = torch.cuda.LongTensor(self.obs) if self.opt.gpu_ids else LongTensor(self.obs)
            weights.index_fill_(1,obs_,weight_val)
            weights=Variable(weights, requires_grad=False)

            self.loss_G_L1 = self.criterionL1(weights * self.fake_B0, weights * self.fake_B0_init.detach()) * self.opt.lambda_C
     
            self.loss_G_L1.backward(retain_graph=True)                
            
        self.fake_B0.backward(pass_grad)

    def backward_G1(self,iter):

        # First, G(A) should fake the discriminator
        if self.opt.conditional:

            fake_AB = torch.cat((self.first_pair.detach(), self.fake_B1), 1)
            pred_fake = self.netD1.forward(fake_AB)
            if self.opt.which_model_preNet != 'none':
                #transform the input
                transformed_AB1 = self.preNet_A.forward(fake_AB)
                pred_fake_GL = self.netD1.forward(transformed_AB1)


            self.loss_G1_GAN = 0
            self.loss_G1_GAN += self.criterionGAN(pred_fake, True)            
        
            if self.opt.which_model_preNet != 'none':
                self.loss_G1_GAN += self.criterionGAN(pred_fake_GL, True)            


        self.loss_G1_L1 = self.criterionL1(self.fake_B1_gt, self.real_B1) * self.opt.lambda_A
        fake_B1_gray = 1-torch.nn.functional.sigmoid(100*(torch.mean(self.fake_B1,dim=1,keepdim=True)-0.9))
        real_A1_gray = 1-torch.nn.functional.sigmoid(100*(torch.mean(self.real_A1,dim=1,keepdim=True)-0.9))
        self.loss_G1_MSE_rgb2gay = self.criterionMSE(fake_B1_gray, real_A1_gray.detach())* self.opt.lambda_A/3.0


        real_A1_gt_gray = 1-torch.nn.functional.sigmoid(100*(torch.mean(self.real_A1_gt,dim=1,keepdim=True)-0.9))
        real_B1_gray = 1-torch.nn.functional.sigmoid(100*(torch.mean(self.real_B1,dim=1,keepdim=True)-0.9))


        self.loss_G1_MSE_gt = self.criterionMSE(real_A1_gt_gray, real_B1_gray)* self.opt.lambda_A
        
        # update generator less frequently
        if iter<200:
            rate_gen = 90
        else:
            rate_gen = 60
        

        if (iter%rate_gen)==0:
            self.loss_G1 = self.loss_G1_GAN + self.loss_G1_L1 + self.loss_G1_MSE_gt
            G1_L1_update = True
            G1_GAN_update = True
        else:
            self.loss_G1 = self.loss_G1_L1 + self.loss_G1_MSE_gt
            G1_L1_update = True
            G1_GAN_update = False

        if (iter<200):
            self.loss_G1 += self.loss_G1_MSE_rgb2gay
        else:
            self.loss_G1 += 0.01*self.loss_G1_MSE_rgb2gay

        

        self.loss_G1.backward(retain_graph=True)

        (b,c,m,n) = self.real_A1_s.size()
        self.real_A1_grad = torch.zeros(b,c,m,n).cuda() if self.opt.gpu_ids else torch.zeros(b,c,m,n)

        
        if G1_L1_update:
            for batch in self.obs:
                self.real_A1_grad[batch,:,:,:] = self.real_A1_gt_s.grad.data[self.id_[batch],:,:,:]


    def optimize_parameters(self,iter):
        self.forward0()
        self.forward1(inp_grad=True)
        self.prepare_data()
        
        if self.opt.which_model_preNet != 'none':
            self.optimizer_preA.zero_grad()
        self.optimizer_D1.zero_grad()
        self.backward_D1()
        self.optimizer_D1.step()
        if self.opt.which_model_preNet != 'none':
            self.optimizer_preA.step()
        self.optimizer_E1.zero_grad()
        self.optimizer_DE1.zero_grad()
        self.backward_G1(iter)
        self.optimizer_DE1.step()
        self.optimizer_E1.step()
        
        self.loss_G_L1 = Variable(torch.zeros(1))


    def optimize_parameters_Stacked(self,iter):
        self.forward0()
        self.forward1(inp_grad=True)
        self.prepare_data()
        
        if self.opt.which_model_preNet != 'none':
            self.optimizer_preA.zero_grad()

        self.optimizer_D1.zero_grad()
        self.backward_D1()
        self.optimizer_D1.step()
        if self.opt.which_model_preNet != 'none':
            self.optimizer_preA.step()
        self.optimizer_E1.zero_grad()
        self.optimizer_DE1.zero_grad()
        self.backward_G1(iter)
        self.optimizer_DE1.step()
        self.optimizer_E1.step()
        
        b,c,m,n = self.fake_B0.size()
        self.optimizer_G.zero_grad()
        if self.opt.conv3d:
            self.optimizer_G_3d.zero_grad()


        b,c,m,n = self.fake_B0.size()

        fake_B0_grad = torch.zeros(b,c,m,n).cuda() if self.opt.gpu_ids else torch.zeros(b,c,m,n)
        real_A_grad = self.real_A1_grad
        
        for batch in range(self.opt.input_nc):
            if not self.opt.rgb_in and self.opt.rgb_out:
                fake_B0_grad[self.id_[batch], batch,:,:] += torch.mean(real_A_grad[batch,:,:,:],0)*3
            else: 
                #TODO  
                fake_B0_grad[batch, self.obs[batch]*np.array(self.opt.input_nc_1):(self.obs[batch]+1)*np.array(self.opt.input_nc_1),:,:] = real_A_grad[batch,:,:,:]

        self.backward_G(fake_B0_grad, iter)
        self.optimizer_G.step()
        if self.opt.conv3d:
            self.optimizer_G_3d.step()


    def get_current_errors(self):
        return OrderedDict([('G1_GAN', self.loss_G1_GAN.data[0]),
                ('G1_L1', self.loss_G1_L1.data[0]),
                ('G1_MSE_gt', self.loss_G1_MSE_gt.data[0]),
                ('G1_MSE', self.loss_G1_MSE_rgb2gay.data[0]),
                ('D1_real', self.loss_D1_real.data[0]),
                ('D1_fake', self.loss_D1_fake.data[0]),
                ('G_L1', self.loss_G_L1.data[0])
        ])


    def get_current_visuals(self):
        real_A1 = self.real_A1.data.clone()
        g,c,m,n = real_A1.size()
        fake_B = self.fake_B1.data.clone()
        real_B = self.real_B1.data.clone()
        
        if self.opt.isTrain:
            real_A_all = real_A1
            fake_B_all = fake_B
        else:
            real_A_all = self.Tensor(real_B.size(0),real_B.size(1),real_A1.size(2),real_A1.size(2)*real_A1.size(0))
            fake_B_all = self.Tensor(real_B.size(0),real_B.size(1),real_A1.size(2),fake_B.size(2)*fake_B.size(0))
            for b in range(g):
                real_A_all[:,:,:,self.out_id[b]*m:m*(self.out_id[b]+1)] = real_A1[b,:,:,:]
                fake_B_all[:,:,:,self.out_id[b]*m:m*(self.out_id[b]+1)] = fake_B[b,:,:,:]

        real_A = util.tensor2im(real_A_all)
        fake_B = util.tensor2im(fake_B_all)
        real_B = util.tensor2im(self.real_B1.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        if not self.opt.no_Style2Glyph:
            try:
                G_label = str(int(label)+int(self.opt.which_epoch))
            except:
                G_label = label
            if self.opt.conv3d:
                self.save_network(self.netG_3d, 'G_3d', G_label, self.gpu_ids)
            self.save_network(self.netG, 'G', G_label, self.gpu_ids)
        self.save_network(self.netE1, 'E1', label, self.gpu_ids)
        self.save_network(self.netDE1, 'DE1', label, self.gpu_ids)
        self.save_network(self.netD1, 'D1', label, self.gpu_ids)
        if self.opt.which_model_preNet != 'none':
            self.save_network(self.preNet_A, 'PRE_A', label, gpu_ids=self.gpu_ids)


    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        if self.opt.which_model_preNet != 'none':
            for param_group in self.optimizer_preA.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_D1.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_E1.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_DE1.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr