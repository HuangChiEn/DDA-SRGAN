#!/usr/bin/env python3
# -*- coding: utf-8 -*-   ## In the case to print chinese character.
import argparse

def get_config():
    ## (1) Parameter setting stage : the parameters used in module are defined by following code : 
    parser = argparse.ArgumentParser(description="The following user defined parameters are recommand to adjustment according to your requests.")

    ## Execution enviroment setting : 
    parser.add_argument('--module_name', type=str, default="MA_SRGAN", help='module name.')
    parser.add_argument('--cuda', type=str, default="0, 1, 2, 3", help='a list of gpus.')
    parser.add_argument('--exe_mode', type=str, default="training", help='execution mode : {training, evaluating, generating}')
    parser.add_argument('--force_cpu', type=bool, default=False, help='Force cpu execute the assignments of model whatever gpu assistance.')
    parser.add_argument('--n_gpus', type=int, default=4, help='num of gpu.')
    
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
    parser.add_argument('--save_dir_name', type=str, default="training/MA_SRGAN/exp002_CASIA_lab", help="The directory name of saving sample images under ../images dir.")
    parser.add_argument('--file_ext', type=str, default="bmp", help="The file extension of generated images.")
    parser.add_argument('--pre_model_dir', type=str, default='../pretrain', help="The directory of pretrain model.")
    parser.add_argument('--exp_tag', type=str, default="002", help="The tag of experiment, {main_ver}_{brnch}_{patch}, ex. 000_1_2, the patch for bug in version 000 with 1 brnch to new functionality.")
    
    ## Generating setting : 
    parser.add_argument('--generate_num', type=int, default=36010, help="The number of lower resolution with generate sr images.")
    parser.add_argument('--lr_img_dir', type=str, default="eval_set", help="The directory name of low resolution set.")
    parser.add_argument('--save_img_dir', type=str, default="generating/MA_SRGAN/exp002_CASIA_lab", help="The directory name of generated images.")
    
    args = parser.parse_args() # parser the arguments, get parameter via arg.parameter

    return args
