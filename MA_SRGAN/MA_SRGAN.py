#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import  ## execute the following code in python3.x enviroment
"""
Created on Tue May  18 15:32:00 2020
@author: Josef-Huang

@@ The following code is stable ' V 2.2 (stable) version '
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
    
    The kernel module be divided into  generator, discriminator, auxiliary_featuretor. The different module 
    can load different type of basic block :  
    (1) generator -> RRDRB(Residual in Residual Dense Residual Block)
        -> The subpixel conv are eliminated(currently), due to their effect are not good~(see problem sample)
        -> However, respect with paper proposed model, I trace back to ECPNetwork.
        (main stream)@->
            @-> I use keras Upsampling2D function to upsampling(nearest), 
                I may attempt to replace it into TransposeConv or SubPixel Conv.
            @-> Due to the network size, I should reduce part of it to implement more
                functionality and accelerate the prediction phase.
                @@-> The 2014 proposed VGG-net should be replace to other structure.
                    The mobile-net v2, v3 are candidate of original VGG.
            @-> The attention mechnism will be include into ESRGAN,
                for generate the data of specific domain task. 
                    
Acknowledegment :
    The following code are referenced from a lot of released source code and unreleased code.
    I want to greatly thank for the author on github who released the srgan.py source code.
    
    When I move the framework of released code into my project, my senior k0 Young, who offer 
    some scheme to build the parallel GPU env, so that I can run the experiments quickly.
    Beside, he found the bug in Resnet code to help me reduce the computation cost.
    
    And my senior sero offer a lot of advice to help me de-very-big-bug with lower cost.
    The learning rate should adjust even in Adam method.
    
    At the result, of course my prof. Liu, who offer the lab to support our research.
    Every research can not be done completely without the env support!!
    
Notice : 
    As the file name, the edsr gan module will be implement in this code.
    The RaGAN loss function will be implement in the code, and the loss weight coefficient 
    with different loss function still request to decide.
    At the present, [1.0 (feature content_loss), 1e-3 (RaDis loss), 1e-3 (L1 norm loss)] loss 
    weight are in use.
    
    At the next stage, i'm going to change the G, auxMod structure with transConv2d and MobileNet, 
    instead of bilinear upsampling method and VGG19.

            Josef-Huang...2020/06/04 (Donnerstag)
"""

import os
from os.path import join
## log_dump_level : 1-Info 2-Warning 3-Error 4-Fatal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

## tensorflow backend--GPU env seeting : 
import tensorflow as tf
from keras.utils import multi_gpu_model

## some related utility : 
import imageio as imgIO
import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../shared_module')

## keras package : the high-level module for deep learning 
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.optimizers import Adam

## setting callback mechnism : 
'''
# 1). FREEUSE
from keras.callbacks import TensorBoard
ensure_dir( './TensorBoard/' + file_name )
path = './TensorBoard/' + file_name + 'logs'  
tensorboard = TensorBoard(log_dir=path, 
                          histogram_freq=0,
                          batch_size=batch_size, 
                          write_graph=True, 
                          write_grads=True, 
                          write_images=True )

# 2). FREEUSE
from keras.callbacks import EarlyStopping
## Early stopping technique : 
##  Not very recommand to use, since the generator may request more epochs to
##      generate high-quilty image in sometimes, 
##               although the loss may stuck in some range. 
##      If you want to set early stopping, at least 500 epochs to be patience.

earlyStop = EarlyStopping(monitor='val_loss', 
                          min_delta=0, 
                          patience=500, 
                          verbose=1 , 
                          mode='auto')

# 3). FREEUSE
##TODO.2 : Adding the checkpoints mechnism.. 
'''

## Self-define package 
from common_interface import SR_base_model

# from dataloader import DataLoader 
from dev_dataloader import Data_loader

from modelManager import modelManager
from submodule import load_module
import losses_package as loss_pkg
#from Image_Generator_Using_Center_Score import ImageGenerator

class MA_SRGAN(SR_base_model):
    
    def __init__(self, lr_height, lr_width, n_D_filt,  
                 n_gpus,  exe_mode, learn_rate, 
                 n_G_filt, img_scalar, n_RRDRB, n_DRB, res_scalar, 
                 **_):  
        
        def train_init():
            # Common Factor : 
            self.loss_weights = {'pixel': 0.01, 'percept': 1, 'atten': 0.01, 'gen': 0.005}
            self.lr = learn_rate
            self.optimizer = Adam(lr=self.lr, beta_1=0.5, beta_2=0.999, amsgrad=True)

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
            D_params = {'num_of_filts' : n_D_filt,
                    'hr_shape' : self.hr_shape}
            self.discriminator = load_module("discriminator", D_params)
            
            ## load attention-based (mask generator) and (mask operator) - 
            self.msk_shape = (self.hr_height, self.hr_width)
            
            # mask operator :
            Attdis_params = {'img_shape' : self.hr_shape,
                    'img_mask_shape' : self.msk_shape} 
            self.atten_dis = load_module("attention_dis", Attdis_params)
            
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
        
        mode_dict = {"training":train_init, "evaluating":eval_init, "generating":generate_init}
        assert (exe_mode in mode_dict.keys()),\
            "KEY ERROR_MESSGAE : The exe_mode should be {}".format(mode_dict.keys())
        print("The {} mode is about to execute..".format(exe_mode))
        
        mode_dict[exe_mode]()   
        
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
        self.atten_dis.trainable = False
        
        ## Image setting for model input structure : 
        img_lr, img_hr, img_mask = Input(shape=self.lr_shape), Input(shape=self.hr_shape), Input(shape=self.msk_shape)
        
        fake_hr = self.generator(img_lr)
        fake_fea = self.feature_dis(fake_hr)
        att_fea = self.atten_dis([fake_hr, img_mask])  ## ROI fea extract
        
        ##  Discriminator determines validity of generated high resolution images 
        dis_real, dis_fake = self.discriminator(img_hr), self.discriminator(fake_hr)
        relative_gen_loss = loss_pkg.custom_rela_gen_loss(dis_real, dis_fake)
        
        ## At the result : build SrGAN with multi-output and loss 
        tmpModel = Model(inputs=[img_lr, img_hr, img_mask], 
                             outputs=[fake_hr, fake_fea, att_fea, 
                                      dis_real, dis_fake], 
                                          name='MA-SRGAN')
        
        model = multi_gpu_model(tmpModel, gpus=self.n_gpus) if self.n_gpus > 1 else tmpModel
        
        model.compile(optimizer=self.optimizer,               
                loss=[loss_pkg.pixel_loss, loss_pkg.perceptual_loss, 
                      loss_pkg.attention_l1_loss, relative_gen_loss, None],               
                loss_weights=[self.loss_weights['pixel'], self.loss_weights['percept'], 
                      self.loss_weights['atten'], self.loss_weights['gen'], 0.])
        
        return model
    
    
    def training(self, data_set_root, data_set_name, file_ext,
                     load_G_D_weight, save_generator, pre_model_dir,
                         samp_img_intval, save_dir_name, exp_tag,
                             batch_size, epochs, sample_file_ext='png', train_set_name='tra_set',
                             **_): 
        
        def sample_images(epoch, save_dir_name, demo_gen):
            
            def save_each_img(imgs_lst, str_lst):
                for imgs, string in zip(imgs_lst, str_lst):
                    for idx, img in enumerate(imgs):
                        fig = plt.figure(num=idx)
                        plt.figure(fig.number)
                        plt.imshow(img)
                        imgs_templt = join(save_dir_name, 
                                             "{0}_{1}{2}.{3}".format(
                                             epoch, string, idx, sample_file_ext))
                        fig.savefig(join(os.pardir, "images", imgs_templt))
                        plt.close(fig)
                    
            os.makedirs(join(os.pardir, "images", save_dir_name), exist_ok=True)
            rows, cols = 2, 2  # for col=2 means save sr, hr img only.
            
            imgs_lr, imgs_hr = next(demo_gen)
            
            fake_hr = self.generator.predict(imgs_lr)
    
            # Rescale images 0 - 1
            rescalr = lambda x: 0.5 * x + 0.5
            imgs_lr, fake_hr, imgs_hr = \
                rescalr(imgs_lr), rescalr(fake_hr), rescalr(imgs_hr)
    
            # Save generated images and the high resolution originals
            titles = ['Generated', 'Original']
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
            
            save_each_img([imgs_lr, fake_hr, imgs_hr], 
                          ["lowres", "super", "original"])
            
            
        self.model_manger = modelManager(pretrain_dir=pre_model_dir, 
                                      model_type=__name__, dataset_name=data_set_name)
        
        data_src_path = join("..", data_set_root, data_set_name, train_set_name)
        self.data_loader = Data_loader(data_src_path, hr_img_size=(self.hr_height, self.hr_width), scalr=self.img_scalar, ext=file_ext)
        
        
        ## load G and D model weights : 
        if load_G_D_weight:
            self.generator, self.discriminator = \
                self.model_manger.load_model_weight(generator=self.generator, 
                                                    discriminator=self.discriminator,
                                                    exp_tag=exp_tag)
            print('Procedure of loading the weights of Gen, Dis modlues is complete..\n')
        
        dummy = np.zeros((batch_size, 1), dtype=np.float32)
        start_time = datetime.datetime.now()
        
        
        data_gen_G = self.data_loader.ld_msk_data_gen(batch_size, fliplr=True, 
                                         msk_path=join(data_src_path), msk_typ='CelebA')
        data_gen_D = self.data_loader.ld_data_gen(batch_size, fliplr=True)
        demo_gen = self.data_loader.ld_data_gen(batch_size, fliplr=True)
        
        for epoch in range(epochs):
            imgs_lr, imgs_hr, imgs_msk = next(data_gen_G)
            
            real_fea = self.feature_dis.predict(imgs_hr)
            ROI_part = self.atten_dis.predict([imgs_hr, imgs_msk])
            
            self.generator.trainable = True
            self.discriminator.trainable = False
            g_loss = self.SrGAN.train_on_batch([imgs_lr, imgs_hr, imgs_msk], [imgs_hr, real_fea, ROI_part, dummy])
            
            imgs_lr, imgs_hr = next(data_gen_D)
            
            self.generator.trainable = False
            self.discriminator.trainable = True
            d_loss = self.RaDis.train_on_batch([imgs_lr, imgs_hr], [dummy])
            
            elapsed_time = datetime.datetime.now() - start_time
            
            print ("Epoch : %d ; time: %s\n" % (epoch, elapsed_time))
            
            ## Evaluation model : 
            print("  RaDis out : \n  D -> {}\n\n".format(d_loss[1]))
            print("  SrGAN out : Total loss -> {}\n".format(g_loss[0]))
            print("    G -> {} ; RaGen -> {}\n".format(g_loss[1], g_loss[4]))
            print("    att -> {}\n".format(g_loss[3]))
            print("    FeaDis -> {}\n".format(g_loss[2]))
            print("\n\n--------------------------------------\n\n")
            
            # If at save interval => save generated image samples
            if epoch % samp_img_intval == 0:
                sample_images(epoch, save_dir_name, demo_gen)
                
        if save_generator:
            ## Build the model config : (optional)
            #  Basic Key - 
            # 'model_type', 'loss_weight', 'learning_rate', 'optimizer', 
            # 'epochs',     'batch_size'.
            model_config_dict = {"model_type":__name__, "loss_weight":self.loss_weights,
                                 "learning_rate":self.lr, "optimizer":self.optimizer.__doc__.split("\n")[0],
                                 "epochs":epochs, "batch_size":batch_size}
            
            self.model_manger.save_model_weight(generator=self.generator, discriminator=self.discriminator, 
                                                exp_tag=exp_tag, model_config_dict=model_config_dict)
            
            
    def generating(self, data_set_root, data_set_name, lr_img_dir, exp_tag,
                       save_img_dir, n_sr_img, file_ext, pre_model_dir,
                       gen_batch_size=1, **_):
        
        def store_img(img, file_name, file_ext='png'):
            # Rescale images to range [0 - 1]
            img = 0.5 * img + 0.5
            imgs_templt = join(os.pardir, "images", save_img_dir, 
                               "{0}.{1}".format(file_name, file_ext))
            imgIO.imwrite(imgs_templt, img)  # (img*255.0).astype(np.uint8)
        
        os.makedirs(join(os.pardir, "images", save_img_dir), exist_ok=True)
        
        self.model_manger =  modelManager(pretrain_dir=pre_model_dir, model_type=__name__, 
                                       dataset_name=data_set_name, exp_tag=exp_tag)
        self.generator, _ = self.model_manger.load_model_weight(generator=self.generator)
        self.generator = multi_gpu_model(self.generator, gpus=2)
        
        data_src_path = join('..', data_set_root, data_set_name, lr_img_dir)
        self.data_loader = Data_loader(data_src_path, hr_img_size=(self.hr_height, self.hr_width), scalr=self.img_scalar, ext=file_ext)
        
        div, mod = divmod(n_sr_img, gen_batch_size)
        assert((mod == 0))  ## Total image should be divided by batch_size exactly
        gen_epoch = div
        
        ## Get the image generator.
        img_gen = self.data_loader.ld_generate_data_gen(gen_batch_size, fliplr=True)
        
        try:
            for _ in range(gen_epoch):
                lr_img, file_name = next(img_gen)
                sr_img = self.generator.predict(lr_img)
               
                for _, img in enumerate(sr_img):
                    store_img(img, file_name)
                del lr_img
                del sr_img
                
        except StopIteration:
            print("The image generating procedure are already complete.\n",\
                      "If the time of complete is shorter than you thought, ",\
                          "you may check the number of data in lower resolution datasets.\n")  
        
        
    def evaluation(self):
        raise NotImplementedError("The evaluation procedure of MA-SRGAN should prograss on bio_eval.")
       
## Notice : For passing the unit test, any information will not represesnt in command when you run the code.
if __name__ == '__main__':
    pass    