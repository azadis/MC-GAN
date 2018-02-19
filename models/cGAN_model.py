################################################################################
# MC-GAN
# Glyph Network Model
# By Samaneh Azadi
################################################################################

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
import random



class cGANModel(BaseModel):
    def name(self):
        return 'cGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        if self.opt.conv3d:
            self.netG_3d = networks.define_G_3d(opt.input_nc, opt.input_nc, norm=opt.norm, groups=opt.grps, gpu_ids=self.gpu_ids)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, opt.use_dropout, gpu_ids=self.gpu_ids)
        
        disc_ch = opt.input_nc
            
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.opt.conditional:
                if opt.which_model_preNet != 'none':
                    self.preNet_A = networks.define_preNet(disc_ch+disc_ch, disc_ch+disc_ch, which_model_preNet=opt.which_model_preNet,norm=opt.norm, gpu_ids=self.gpu_ids)
                nif = disc_ch+disc_ch

                
                netD_norm = opt.norm

                self.netD = networks.define_D(nif, opt.ndf,
                                             opt.which_model_netD,
                                             opt.n_layers_D, netD_norm, use_sigmoid, gpu_ids=self.gpu_ids)

            else:
                self.netD = networks.define_D(disc_ch, opt.ndf,
                                             opt.which_model_netD,
                                             opt.n_layers_D, opt.norm, use_sigmoid, gpu_ids=self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            if self.opt.conv3d:
                 self.load_network(self.netG_3d, 'G_3d', opt.which_epoch)
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                if opt.which_model_preNet != 'none':
                    self.load_network(self.preNet_A, 'PRE_A', opt.which_epoch)
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()


            # initialize optimizers
            if self.opt.conv3d:
                 self.optimizer_G_3d = torch.optim.Adam(self.netG_3d.parameters(),
                                                     lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.which_model_preNet != 'none':
                self.optimizer_preA = torch.optim.Adam(self.preNet_A.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            if self.opt.conv3d:
                networks.print_network(self.netG_3d)
            networks.print_network(self.netG)
            if opt.which_model_preNet != 'none':
                networks.print_network(self.preNet_A)
            networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']        
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        if self.opt.conv3d:
            self.real_A_indep = self.netG_3d.forward(self.real_A.unsqueeze(2))
            self.fake_B = self.netG.forward(self.real_A_indep.squeeze(2))
        else:
            self.fake_B = self.netG.forward(self.real_A)

        
        self.real_B = Variable(self.input_B)
        real_B = util.tensor2im(self.real_B.data)
        real_A = util.tensor2im(self.real_A.data)
    
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
        self.real_A = Variable(self.input_A, volatile=True)
        if self.opt.conv3d:
            self.real_A_indep = self.netG_3d.forward(self.real_A.unsqueeze(2))
            self.fake_B = self.netG.forward(self.real_A_indep.squeeze(2))

        else:
            self.fake_B = self.netG.forward(self.real_A)
            
        self.real_B = Variable(self.input_B, volatile=True)

    #get image paths
    def get_image_paths(self):
        return self.image_paths

    
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        label_fake = self.add_noise_disc(False)

        b,c,m,n = self.fake_B.size()
        rgb = 3 if self.opt.rgb else 1

        self.fake_B_reshaped = self.fake_B
        self.real_A_reshaped = self.real_A
        self.real_B_reshaped = self.real_B

        if self.opt.conditional:


            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A_reshaped, self.fake_B_reshaped), 1))
            self.pred_fake_patch = self.netD.forward(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(self.pred_fake_patch, label_fake)
            if self.opt.which_model_preNet != 'none':
                #transform the input
                transformed_AB = self.preNet_A.forward(fake_AB.detach())
                self.pred_fake = self.netD.forward(transformed_AB)
                self.loss_D_fake += self.criterionGAN(self.pred_fake, label_fake)
                            
        else:
            self.pred_fake = self.netD.forward(self.fake_B.detach())
            self.loss_D_fake = self.criterionGAN(self.pred_fake, label_fake)

        # Real
        label_real = self.add_noise_disc(True)
        if self.opt.conditional:
            real_AB = torch.cat((self.real_A_reshaped, self.real_B_reshaped), 1)#.detach()
            self.pred_real_patch = self.netD.forward(real_AB)
            self.loss_D_real = self.criterionGAN(self.pred_real_patch, label_real)

            if self.opt.which_model_preNet != 'none':
                #transform the input
                transformed_A_real = self.preNet_A.forward(real_AB)
                self.pred_real = self.netD.forward(transformed_A_real)
                self.loss_D_real += self.criterionGAN(self.pred_real, label_real)
                            
        else:
            self.pred_real = self.netD.forward(self.real_B)            
            self.loss_D_real = self.criterionGAN(self.pred_real, label_real)
        
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()


    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.opt.conditional:
            #PATCH GAN
            fake_AB = (torch.cat((self.real_A_reshaped, self.fake_B_reshaped), 1))
            pred_fake_patch = self.netD.forward(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake_patch, True)
            if self.opt.which_model_preNet != 'none':
                #global disc
                transformed_A = self.preNet_A.forward(fake_AB)
                pred_fake = self.netD.forward(transformed_A)
                self.loss_G_GAN += self.criterionGAN(pred_fake, True)
        
        else:
            pred_fake = self.netD.forward(self.fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        if self.opt.which_model_preNet != 'none':
            self.optimizer_preA.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        if self.opt.which_model_preNet != 'none':
            self.optimizer_preA.step()


        self.optimizer_G.zero_grad()
        if self.opt.conv3d:
            self.optimizer_G_3d.zero_grad()

        self.backward_G()
        self.optimizer_G.step()
        if self.opt.conv3d:
            self.optimizer_G_3d.step()
        
    

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                ('G_L1', self.loss_G_L1.data[0]),
                ('D_real', self.loss_D_real.data[0]),
                ('D_fake', self.loss_D_fake.data[0])
        ])


    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        if self.opt.conv3d:
             self.save_network(self.netG_3d, 'G_3d', label, gpu_ids=self.gpu_ids)
        self.save_network(self.netG, 'G', label, gpu_ids=self.gpu_ids)
        self.save_network(self.netD, 'D', label, gpu_ids=self.gpu_ids)
        if self.opt.which_model_preNet != 'none':
            self.save_network(self.preNet_A, 'PRE_A', label, gpu_ids=self.gpu_ids)
            

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        if self.opt.which_model_preNet != 'none':
            for param_group in self.optimizer_preA.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.conv3d:
            for param_group in self.optimizer_G_3d.param_groups:
                param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr