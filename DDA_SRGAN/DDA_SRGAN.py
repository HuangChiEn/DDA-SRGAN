#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import  ## execute the following code in python3.x enviroment
"""
Created on Tue May  18 15:32:00 2020
@author: Josef-Huang

@@ The following code is stable ' V 3-alpha (unstable) version '
Signature :
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@                                                                  @@
@@    JJJJJJJJJ    OOOO     SSSSSSS   EEEEEE   PPPPPPP   HH    HH   @@
@@       JJ       O    O    SSS       E        PP   PP   HH    HH   @@
@@       JJ      O      O    SSS      EEEEEE   PPPPPPP   HHHHHHHH   @@
@@       JJ       O    O         SS   E        PP        HH    HH   @@
@@     JJJ         OOOO     SSSSSSS   EEEEEE   PP        HH    HH   @@
@@                                                                  @@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    

Description :
    The following code is the implementation of super resolution GAN by keras mainly with tensorflow backend
    engining.
    
    The kernel module is mainly divided into generator, discriminator and feature_discriminator. 
    The different module can load different type of basic block :  
    (1) generator -> RRDRB(Residual in Residual Dense Residual Block)
        -> The subpixel conv are eliminated(permanatly), due to their effect are not good~(see problem sample)
        -> For now, bicubic convolution upsampling is in used.
        (main stream)@->
            @-> Due to the network size, I should reduce part of it to implement more
                functionality and accelerate the prediction phase.
            @-> The channel attention as well as the spatial attention mechanism will be included 
                into nESRGAN+ framework, for forcing the generator focus on generating the 
                high quality result in ROI part. 
                    
Acknowledegment :
    The following code are referenced from a lot of released source code and unreleased code.
    I want to greatly thank for all author on github.
    
    When I move the framework of released code into my project, my senior k0 Young, who offer 
    some scheme to build the parallel GPU env, so that I can run the experiments quickly.
    Beside, he found the bug in Resnet code to help me reduce the computation cost.
    
    And my senior sero offer a lot of advice to help me debug the lower cost.
    The learning rate should be adjusted even in Adam method 
    (no wonder darknet author commented Adam is suck...).
    
    At the result, of course my prof. Li, who offer the lab to support our research.
    Every research can not be done completely without the env support!!


Author :
            Josef-Huang...03/22/2021 (Montag)
"""

## tensorflow backend--GPU env seeting : 
from keras.utils import multi_gpu_model
import PIL.Image as pil_image
import os
from os.path import join
## log_dump_level : 1-Info 2-Warning 3-Error 4-Fatal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

## some related utility : 
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../shared_module')

## keras package : the high-level module for deep learning 
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam

## Self-define package 
from common_interface import SR_base_model

#from dataloader import DataLoader  
from dev_dataloader import Data_loader

from modelManager import modelManager
from submodule import load_module    
import losses_package as loss_pkg

## Spatial weighted Channel Attention GAN
class SWCA_GAN(SR_base_model):
    
    def __init__(self, lr_height, lr_width, n_D_filt,  
                 n_gpus,  exe_mode,  
                 learn_rate, n_G_filt, img_scalar, 
                 n_RRDRB, n_DRB, res_scalar, 
                 **_):  
                           #  now --> { 'pixel':4e-1, 'percept':1.3,  'gen':8e-2 }
        def train_init():  # face { 'pixel':1e-2, 'percept':1,  'gen':5e-2 }
            self.loss_weights = { 'pixel':5e-1, 'percept':1.3,  'gen':1e-1 }  ## face -- ('pixel':5e-1, 'percept':7e-1,  'gen':7e-2 ) ; iris -- 'pixel':5e-2, 'percept':1,  'gen':3e-2 
            self.optimizer = Adam(lr=learn_rate, beta_1=0.5, beta_2=0.999, amsgrad=True)  # beta1=0.5
            self.lr = learn_rate
            ## (1) Load Basic Network Components : 
            
            ## load generator - 
            #   parameter setting 
            G_params = {
                    ## self-define :
                    'num_of_filts' : n_G_filt,
                    'num_of_RRDRB' : n_RRDRB,
                    'lr_shape' : self.lr_shape,
                    ## default :
                    'num_of_DRB' : n_DRB,
                    'upScalar' : self.img_scalar,
                    'resScal' : res_scalar
                    }
            self.generator = load_module("generator", G_params)
            
            ## load feature_discriminator - 
            self.feature_dis = load_module("feature_dis", {'hr_shape' : self.hr_shape})
            
            ## load discriminator - 
            self.discriminator = load_module("discriminator", 
                                             {'num_of_filts' : n_D_filt, 'hr_shape' : self.hr_shape})
            
        
            ## (2) Build high-level inner modules : 
            ## build RaGAN -  by previous defined basic network component.
            self.RaDis = self.__inner_build_RaGAN()
            
            ## build SrGAN - (main module, need compile)
            self.SrGAN = self.__inner_build_SrGAN()
            
               
        def generate_init():
            ## Configure model manager :
            G_params = {
                    ## self-define :
                    'num_of_filts' : n_G_filt,
                    'num_of_RRDRB' : n_RRDRB,
                    'lr_shape' : self.lr_shape,
                    ## default :
                    'num_of_DRB' : n_DRB,
                    'upScalar' : self.img_scalar,
                    'resScal' : res_scalar
                    }
            self.generator = load_module("generator", G_params)
            
        def eval_init():
            raise NotImplementedError("The evaluation procedure of MA-SRGAN should prograss on iris recognition.")
         
        
        ## Image input structure : 
        self.lr_shape = self.lr_height, self.lr_width, self.channels = lr_height, lr_width, 3
        self.img_scalar = img_scalar
        self.hr_height, self.hr_width, _ = [dim*img_scalar for dim in self.lr_shape]
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        
        self.n_gpus = n_gpus
        mode_dict = {"training":train_init, "generating":generate_init, "evaluating":eval_init}
        assert (exe_mode in mode_dict.keys()),\
            "KEY ERROR_MESSGAE : The exe_mode should be {}".format(mode_dict.keys())
        print("The {} mode is about to execute..".format(exe_mode))
        self.exe_mode = exe_mode
        
        mode_dict[self.exe_mode]()   
    
    
    def __inner_build_RaGAN(self):
        self.generator.trainable = False
        self.discriminator.trainable = True
        
        ## define discriminator judge part :
        img_lr, img_hr = Input(shape=self.lr_shape), Input(shape=self.hr_shape)
        fake_hr = self.generator(img_lr)
        dis_real, dis_fake = self.discriminator(img_hr), self.discriminator(fake_hr)
        relative_dis_loss = loss_pkg.custom_rela_dis_loss(dis_real, dis_fake)
        model = Model(inputs=[img_lr, img_hr], outputs=[dis_real, dis_fake])
        model.compile(optimizer=self.optimizer, loss=[relative_dis_loss, None], loss_weights=[1.0, 0])
        return model
    
        
    def __inner_build_SrGAN(self):
        ## Trainable setting for properly update G and D parameters : 
        self.generator.trainable = True        ##  Only train G during the SrGAN.train_on_batch phase
        self.discriminator.trainable = False
        self.feature_dis.trainable = False
        
        ## Image setting for model input structure : 
        img_lr, img_hr = Input(shape=self.lr_shape), Input(shape=self.hr_shape)
        
        fake_hr = self.generator(img_lr)
        fake_fea = self.feature_dis(fake_hr)
        
        ##  Discriminator determines validity of generated high resolution images 
        dis_real, dis_fake = self.discriminator(img_hr), self.discriminator(fake_hr)
        relative_gen_loss = loss_pkg.custom_rela_gen_loss(dis_real, dis_fake)
        
        ## At the result : build SrGAN with multi-output and loss 
        tmpModel = Model(inputs=[img_lr, img_hr], 
                            outputs=[fake_hr, fake_fea, dis_real, dis_fake], 
                                    name='SWCA_GAN')
        
        model = multi_gpu_model(tmpModel, gpus=self.n_gpus) if self.n_gpus > 1 else tmpModel
        
        model.compile(optimizer=self.optimizer, 
                loss=[loss_pkg.pixel_loss, loss_pkg.perceptual_loss, 
                      relative_gen_loss, None], 
                loss_weights=[self.loss_weights['pixel'], self.loss_weights['percept'], 
                              self.loss_weights['gen'], 0.])
        
        return model
    
    
    def training(self, data_set_root, data_set_name, file_ext,
                     load_G_D_weight, save_generator, pre_model_dir, exp_tag,
                         samp_img_intval, save_dir_name, 
                             batch_size, epochs, train_set_name="tra_set", save_ckpt=False,
                             **_): 
        
        def sample_images(epoch, save_dir_name, demo_gen):
            
            def save_each_img(imgs_lst, str_lst, lim=2):
                for imgs, string in zip(imgs_lst, str_lst):
                    for idx, img in enumerate(imgs):
                        if idx < lim:
                            fig = plt.figure(num=idx)
                            plt.figure(fig.number)
                            plt.imshow(img)
                            imgs_templt = join(save_dir_name, 
                                                 "{0}_{1}{2}.{3}".format(
                                                 epoch, string, idx, "png"))
                            
                            fig.savefig(join(os.pardir, "images", imgs_templt))
                            plt.close(fig)
            
            os.makedirs(join(os.pardir, "images", save_dir_name), exist_ok=True)
            imgs_lr, imgs_hr = next(demo_gen)
            
            fake_hr = self.generator.predict(imgs_lr)
            fake_hr = np.clip(fake_hr, -1, 1)
            
            # inverted normalization from [-1, 1] into images 0 - 1
            inv_norm = lambda x: 0.5 * x + 0.5
            imgs_lr, fake_hr, imgs_hr = \
                inv_norm(imgs_lr), inv_norm(fake_hr), inv_norm(imgs_hr)
    
            # Save all image in the checkboard style (2 x 2)
            # for col=2 means save sr, hr img only.
            titles = ['Generated', 'Original']
            rows, cols = 2, 2  
            fig, axs = plt.subplots(rows, cols)
            for row in range(rows):
                for col, image in enumerate([fake_hr, imgs_hr]):
                    axs[row, col].imshow(image[row])
                    axs[row, col].set_title(titles[col]) 
                    axs[row, col].axis('off')
            save_templt = join(os.pardir, "images", 
                               save_dir_name, "{}.png".format(epoch))
            fig.savefig(save_templt)
            plt.close()
            
            # save each image with the single image style (1 x 1)
            save_each_img([imgs_lr, fake_hr, imgs_hr], 
                          ["lowres", "super", "original"])
            
            
        assert(self.exe_mode == "training"), \
                "The model was not initialized as training, but for {}\n.".format(self.exe_mode)    
            
        self.model_manger =  modelManager(pretrain_dir=pre_model_dir, model_type=__name__, 
                                       dataset_name=data_set_name, exp_tag=exp_tag)
        ## load G and D model weights : 
        if load_G_D_weight:
            self.generator, self.discriminator = \
                self.model_manger.load_model_weight(generator=self.generator, 
                                                    discriminator=self.discriminator)
            print('Procedure of loading the weights of Gen, Dis modlues is complete..\n')
        
        # config data loader
        data_src_path = join("..", data_set_root, data_set_name, train_set_name)
        self.data_loader = Data_loader(data_src_path, hr_img_size=(self.hr_height, self.hr_width), scalr=self.img_scalar, ext=file_ext)
        # prepare data for the Generator, Discriminator and sampling procedure
        data_gen_G = self.data_loader.ld_data_gen(batch_size, fliplr=True)
        data_gen_D = self.data_loader.ld_data_gen(batch_size, fliplr=True)
        demo_gen = self.data_loader.ld_data_gen(batch_size, fliplr=True)
        
        # dummy GT for Discriminator of RaGAN
        dummy = np.zeros((batch_size, 1), dtype=np.float32)
        # setting max prev_loss for check point mechanism
        prev_loss = math.inf
        
        for epoch in range(epochs):
            start_time = datetime.datetime.now()
            
            ## Generator training phase : 
            imgs_lr, imgs_hr = next(data_gen_G)
            real_fea = self.feature_dis.predict(imgs_hr)
            
            self.generator.trainable = True
            self.discriminator.trainable = False
            
            g_loss = self.SrGAN.train_on_batch([imgs_lr, imgs_hr], [imgs_hr, real_fea, dummy])
            
            ## Discriminator training phase :
            imgs_lr, imgs_hr = next(data_gen_D)
            
            self.generator.trainable = False
            self.discriminator.trainable = True
            d_loss = self.RaDis.train_on_batch([imgs_lr, imgs_hr], [dummy])
            
            elapsed_time = datetime.datetime.now() - start_time
            
            print ("Epoch : %d ; time: %s\n" % (epoch, elapsed_time))
            
            ## Evaluation model :  
            print("  RaDis out : \n  D -> {}\n\n".format(d_loss[1]))
            print("  SrGAN out : Total loss -> {}\n".format(g_loss[0]))
            print("    G -> {} ; RaGen -> {}\n".format(g_loss[1], g_loss[3])) 
            print("    FeaDis -> {}\n".format(g_loss[2]))
            print("\n\n--------------------------------------\n\n")
            
            ## Tracing 'intermediate' result of models during the training
            # save chk-point
            if save_ckpt and g_loss[0] < prev_loss:
                prev_loss = total_loss = g_loss[0]
                print("save check point with loss value : {}".format(total_loss))
                self.model_manger.save_ckpt(generator=self.generator, gen_loss_val=total_loss)
            
            # If at save interval => save generated image samples
            if epoch % samp_img_intval == 0:
                sample_images(epoch, save_dir_name, demo_gen)
                
        # training complete, save models
        if save_generator:
            model_config_dict = {"model_type":__name__, "loss_weight":self.loss_weights,
                                 "learning_rate":self.lr, "optimizer":self.optimizer.__doc__.split("\n")[0],
                                 "epochs":epochs, "batch_size":batch_size}
            
            self.model_manger.save_model_weight(generator=self.generator, discriminator=self.discriminator, 
                                                model_config_dict=model_config_dict)
            
            
            
    def generating(self, data_set_root, data_set_name, lr_img_dir, 
                       save_img_dir, n_sr_img, file_ext, 
                       pre_model_dir, exp_tag, gen_batch_size=1, **_):
        
        def store_img(img, file_name, file_ext='png'):
            raw_img = img
            # Rescale images to range [0 - 1]
            norm_img = np.clip(raw_img, a_min=-1, a_max=1)
            tmp = 0.5 * norm_img + 0.5
            
            imgs_templt = join(os.pardir, "images", save_img_dir, 
                               "{0}.{1}".format(file_name, file_ext))
            
            #result = np.empty_like(tmp, dtype=np.uint8)
            #np.clip(tmp, 0, 255, out=result)
            
            u_img = np.uint8(tmp*255.0)
            nd_img = pil_image.fromarray(u_img)
            nd_img.save(imgs_templt)
            
            
        assert(self.exe_mode == "generating"), \
                "The model was not initialized as generating, but for {}\n.".format(self.exe_mode)    
        
        # valid check for parameters
        os.makedirs(join(os.pardir, "images", save_img_dir), exist_ok=True)
        div, mod = divmod(n_sr_img, gen_batch_size)
        assert((mod == 0))   # Total image should be divided by batch_size exactly
        gen_epoch = div
        
        # prepare generating model
        self.model_manger =  modelManager(pretrain_dir=pre_model_dir, model_type=__name__, 
                                       dataset_name=data_set_name, exp_tag=exp_tag)
        self.generator, _ = self.model_manger.load_model_weight(generator=self.generator)
        self.generator = multi_gpu_model(self.generator, gpus=2)
        
        # prepare input data
        data_src_path = join("..", data_set_root, data_set_name, lr_img_dir)
        self.data_loader = Data_loader(data_src_path, hr_img_size=(self.hr_height, self.hr_width), scalr=self.img_scalar)
        ## Get the image generator.
        img_gen = self.data_loader.ld_generate_data_gen(gen_batch_size, fliplr=True)
        
   
        START = datetime.datetime.now()
        for CNT in range(gen_epoch):
            ## HACKME0.1 : read the lower resolution image by batch
            lr_img, file_name = next(img_gen)
            sr_img = self.generator.predict(lr_img)
            
            for _, img in enumerate(sr_img):
                store_img(img, file_name, file_ext)
            del lr_img
            del sr_img
        
        print((datetime.datetime.now() - START) / (CNT+1))  # CNT begin from zero.
        
        
    def evaluation(self):
        raise NotImplementedError("The evaluation procedure of SWCA_GAN should prograss on iris recognition.")
       
## Notice : For passing the unit test, any information will not represesnt in command when you run the code.
if __name__ == '__main__':
    pass    