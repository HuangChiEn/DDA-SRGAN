#!/usr/bin/env python3
# -*- coding: utf-8 -*-   ## In the case to print chinese character.
## execute the following code in python3.x enviroment, part of python2.x code is not allowed.
from __future__ import print_function, absolute_import  

"""
Created on Dec  16  2020
@author: Josef-Huang

@@ The following code is stable ' V 3.0 (stable version) '

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
    The following code is the implementation of super resolution GAN by keras mainly with 
    tensorflow backend engining.
    And all the assignments of model (learning, generating, etc.) can be declared in here.
            
Acknowledegment :
    The following code are referenced from a lot of released source code and unreleased code.
    I want to greatly thank for the author on github who released the srgan.py source code.
    
    When I move the framework of released code into my project, my senior k0 Young, who offer 
    some scheme to build the parallel GPU env, so that I can run the experiments quickly.
    Beside, he found the bug in Resnet code to help me reduce the computation cost.
    
    And my senior sero offer a lot of advice to help me debug with slightly painful.
    The learning rate should "self-adjust" even the Adam optimizer in used.
    
    At the result, of course my prof. Liu, who offer the lab to support our research.
    Every research can not be done completely without the env support!!

Copy Right : 
    Except for author name representation, all right will be released as source code.
              Josef-Huang...2020/12/16 (Donnerstag)

#########################################################################################
Note : 
    1. Although multi-GPUs is in used, the strategy of distributing the memory 
       may not be balance, due to the main part of gradient of each layer in
       all model's(copies by multi_gpu_model) will be stored in just /:gpu0.
       
       ref : https://github.com/aurotripathy/medium-tf-mem-usage/blob/master/mnist-classifier-colocate-with-gradients.py
       ref : https://github.com/aurotripathy/medium-tf-mem-usage/blob/master/mnist-classifier-skewed-gradient-alloc.py
              
"""

## for import tensorflow backend--GPU env seeting : 
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF  
from os import environ, system, pardir
from os.path import join
## log_dump_level : 1-Info 2-Warning 3-Error 4-Fatal
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

## some related utility : 
import argparse  # User-define parameter setting
import sys       # Grabe the path of modules
sys.path.append('../shared_module')  
sys.path.append('../SWCA_GAN') 

## self-define package : 
from SWCA_GAN import SWCA_GAN         

## Benchmark GAN for SR issue -
#from n_ESRGAN_p import n_ESRGAN_p       ## noise ESRGAN plus.
#from MA_SRGAN import MA_SRGAN           ## Mask Attention mechnaism.

## Main function : ( Declare your parameter setting and model assignments)  
def main():
    ## (1) Parameter setting stage : the parameters used in module are defined by following code : 
    parser = argparse.ArgumentParser(description="The following user defined parameters are recommand to adjustment according to your requests.")

    ## Execution enviroment setting : 
    parser.add_argument('--cuda', type=str, default="0, 1, 2, 3", help='a list of gpus.')
    parser.add_argument('--exe_mode', type=str, default="training", help='execution mode : {training, evaluating, generating}')
    parser.add_argument('--force_cpu', type=bool, default=False, help='Force cpu execute the assignments of model whatever gpu assistance.')
    
    ## Image specification setting :   CelebA (H-50, W-40) ; CASIA_lab (H-120, W-160)
    parser.add_argument('--lr_height', type=int, default=120, help="The height of lower resoultion input image (pixel value).")  
    parser.add_argument('--lr_width', type=int, default=160, help="The width of lower resoultion input image (pixel value).")
    parser.add_argument('--img_scalar', type=int, default=4, help="The size of image scalar (from low resolution to super resolution).")
    
    ## Model specification :  
    parser.add_argument('--n_RRDRB', type=int, default=3, help="The number of residual in residual dense block(RRDB) in the part of Generator.")
    parser.add_argument('--n_G_filt', type=int, default=32, help="The number of filter in Generator.")
    parser.add_argument('--n_D_filt', type=int, default=32, help="The number of filter in Discriminator.")
    ## HACKME.1 : The Patch Discriminator is offline, you can do better.
    #parser.add_argument('--DPatSiz', type=int, default=8, help="The patch size of discriminator (see comment patch GAN).")
    
    ## Learning setting (*): 
    parser.add_argument('--data_set_root', type=str, default="datasets", help="The root of directory of the data set.")
    parser.add_argument('--data_set_name', type=str, default="CASIA_lab", help="The name of dataset, Iris - {CASIA_lab, IOM}, Face - {CelebA}")
    parser.add_argument('--save_generator', type=bool, default=True, help="Save the generator to generate super resolution image in image generation procedure.")
    parser.add_argument('--load_G_D_weight', type=bool, default=False, help="Load the weights of Generator and Discriminator to continue previous training.")
    parser.add_argument('--batch_size', type=int, default=4, help="The number of batch size during training.")
    parser.add_argument('--epochs', type=int, default=20000, help="The number of epoch during training.")
    parser.add_argument('--samp_img_intval', type=int, default=100, help="The interval of sampling the image during training epoch.")
    parser.add_argument('--save_dir_name', type=str, default="training/SWCA_GAN/exp002_CASIA_lab", help="The directory name of saving sample images under ../images dir.")
    parser.add_argument('--file_ext', type=str, default="bmp", help="The file extension of generated images.")
    parser.add_argument('--pre_model_dir', type=str, default='../pretrain', help="The directory of pretrain model.")
    parser.add_argument('--exp_tag', type=str, default="002", help="The tag of experiment, {main_ver}_{brnch}_{patch}, ex. 000_1_2, the patch for bug in version 000 with 1 brnch to new functionality.")
    
    ## Generating setting : 
    parser.add_argument('--generate_num', type=int, default=36010, help="The number of lower resolution with generate sr images.")
    parser.add_argument('--lr_img_dir', type=str, default="eval_set", help="The directory name of low resolution set.")
    parser.add_argument('--save_img_dir', type=str, default="generating/SWCA_GAN/exp002_CASIA_lab", help="The directory name of generated images.")
    
    args = parser.parse_args() # parser the arguments, get parameter via arg.parameter


    ## (2) GPU enviroment setting stage : 
    if args.cuda is not None:
        environ['CUDA_VISIBLE_DEVICES'] = args.cuda   ## Allow the env detect your GPUs.
        system('echo $CUDA_VISIBLE_DEVICES')
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True   # Avoid to run out all memory, allocate it depend on the requiring
        sess = tf.Session(config=config)
        KTF.set_session(sess)
        n_gpus = len(args.cuda.split(','))
        
    else:
        print('Training without gpu. It is recommended using at least one gpu.')
        if args.force_cpu:
            print('Force CPU execution mode.')
            n_gpus = 0
            environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
            environ["CUDA_VISIBLE_DEVICES"] = '-1'
        else:
            raise Exception('sorry, but without gpu without training, since the memory is not enough.')

    ## (3) Initialize parameter list stage : 
        ## The parameters defined in list specifically will not be recommand to modification,
        ##      otherwise you know what you're going to do.
    init_params = {
        ## Execution enviroment setting :
        'n_gpus': n_gpus,
        'exe_mode': args.exe_mode,
        
        ## Image related setting :
        'lr_shape': (args.lr_height, args.lr_width, 3), # RGB
        'img_scalar': args.img_scalar,
            
        ## Model specification :
        'res_scalar':0.2,
        'n_RRDRB': args.n_RRDRB,
        'n_DRB':3, 
        'n_G_filt': args.n_G_filt,
        'n_D_filt': args.n_D_filt,
        #'D_patch_size': args.DPatSiz,  ## see HACKME.1/
        #-----------------------------------------------------------------#
        
        ## Learning setting :
        'learn_rate': 1e-4,        
        'loss_weights': { 'pixel':1e-2, 'percept':1,  'gen':5e-3 }##{'pixel':1e-2, 'percept':1, 'gen':5e-3 }##{ 'pixel':7e-1, 'percept':5e-1,  'atten':5e-2, 'textural':3e-2, 'gen':1e-2 }
    }   
    
    train_params = {  
        #'data_src'
        'data_set_root':join(pardir, args.data_set_root),
        'data_set_name':args.data_set_name,
        'train_set_name':'tra_set',
        'file_ext':args.file_ext,
        'pre_model_dir': args.pre_model_dir,
        'save_generator' : args.save_generator,
        'load_G_D_weight': args.load_G_D_weight,
        'batch_size' : args.batch_size,
        'epochs' : args.epochs,
        'sample_interval' : args.samp_img_intval,
        'save_dir_name':args.save_dir_name,
        'sample_file_ext': 'png',    
        'exp_tag':args.exp_tag,           
        #-----------------------------------------#
        'train_G_ratio' : 1  ## The parameter proposed from Generator part in RGAN implementation.
    }
    
    gen_params = {
        ## Generating setting : 
        'data_set_root':join(pardir, args.data_set_root),
        'data_set_name':args.data_set_name,
        'lr_img_dir':args.lr_img_dir,
        'n_sr_img':args.generate_num,
        'save_img_dir':args.save_img_dir,
        'pre_model_dir': args.pre_model_dir,
        'exp_tag':args.exp_tag,
        'gen_batch_size':1,  ## batch size of generating sr imgs, too big may cause OOM (Out Of Memory) error .
        'file_ext':args.file_ext
    }
    
    ## (4) Assignments of model executing stage : 
    deepMod = SWCA_GAN(**init_params)
    deepMod.training(**train_params)
    #deepMod.generating_img(**gen_params)  

if __name__ == '__main__':
    # Interface for compitable with the older version to execute the code, 
    # plz use the console.py instead.
    main()