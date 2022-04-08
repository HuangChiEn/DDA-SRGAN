#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:26:55 2020

@author: from github
"""
import scipy
import PIL
import os
import re
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio
import tensorflow as tf
import scipy.io as siog
from os.path import join, pardir
from keras.utils import np_utils


class DataLoader():    
    ## TODO : hardcode landmark_loc request flexable paramter setting..
    def __init__(self, data_set_root=None, data_set_name=None, 
                 sub_set_name=None, hr_img_size=(128, 128), scalr=4):
        self.data_src_path = join(pardir, data_set_root, data_set_name, sub_set_name)
        self.landmrk_loc = join(pardir, data_set_root, data_set_name, "Anno", "list_landmarks_align_celeba.txt")
        self.hr_img_size = hr_img_size
        self.msk_shape = self.hr_img_size
        self.scalr = scalr
    
    ##  For loading the training data
    def ld_data_gen(self, batch_size=1, fliplr=False, shuffled=True, 
                                     ext=".bmp", msk_ld_key="CASIA_lab"):
    
        def load_landmark():
            ## Generate the face landmark : 
            landmrk_dict = {}
            
            with open(self.landmrk_loc) as f_ptr:
                f_ptr.readline() ; f_ptr.readline() ## get rid of reduandant info.
                
                for lin in f_ptr:
                    cords_lst = lin.strip().split()
                    filNam = cords_lst.pop(0)
                    ## p1 : (lefteye_x lefteye_y), p2 : (righteye_x righteye_y),
                    ## p3 : (nose_x nose_y), p4 : (leftmouth_x leftmouth_y),
                    ## p5 : (rightmouth_x rightmouth_y)
                    lftEye = (cords_lst[0], cords_lst[1]) ; rgtEye = (cords_lst[2], cords_lst[3])
                    lftMou = (cords_lst[6], cords_lst[7]) ; rgtMou = (cords_lst[8], cords_lst[9])
                    nose = (cords_lst[4], cords_lst[5])
                    landmrk_dict[filNam] = [lftEye, rgtEye, lftMou, rgtMou, nose]
                    
            return landmrk_dict
       
        def get_iris_mask(path, *_):
            path_lst = path.split(os.sep)
            ## Add the mask label in the filename
            file_name_lst = path_lst[-1].split('.')
            mask_name = file_name_lst[-2] + "_mask." + file_name_lst[-1]
            ## joint back to the path from lst.
            mask_dir = os.sep.join(path_lst[:-1])
            mask_path = join(mask_dir, "msk", mask_name)
            return imageio.imread(mask_path)
        
        def get_face_mask(path, landmrk_dict):
            # Draw the white point(pnt) facial landmark with the margin,
            #      on the black map.
            def cords2msk(cords, margin=15):
                landmrk_msk = np.zeros(self.hr_img_size)
                
                for pnt in cords:
                    pnt = [int(pnt[0]), int(pnt[1])]
                    pnt = np.array(pnt)
                    up_pnt = pnt + margin
                    dwn_pnt = pnt - margin 
                    landmrk_msk[dwn_pnt[0]:up_pnt[0]+1, dwn_pnt[1]:up_pnt[1]+1] = 1.0
                    
                return landmrk_msk 
            
            path_lst = path.split(os.sep)
            file_name = path_lst.pop(-1)
                
            cords = landmrk_dict[file_name]
            landmrk_msk = cords2msk(cords)
                
            return landmrk_msk
        
    
        def get_batch_img(imgs_path, mask_ld_dict):
            
            def iter_path(path_lst):
                while True:
                    for path in path_lst:
                        yield path
                    
                
            ld_mask_func = mask_ld_dict[msk_ld_key]
            landmark_dict = load_landmark() if msk_ld_key == "CelebA" else [-1]
            
            iteration = len(imgs_path)//batch_size
            itr_obj = iter_path(imgs_path)
            [h, w] = self.hr_img_size
            low_h, low_w = (h // self.scalr), (w // self.scalr)
            
            
            for _ in range(iteration):  ## load one img per next()
                hr_imgs, lr_imgs, msk_imgs = [], [], []
                for idx in range(batch_size):
                    path = next(itr_obj)
                        
                    ## load & downsampling img
                    raw_hr, img_hr = PIL.Image.open(path), np.asarray(PIL.Image.open(path), 'f')
                    img_lr = np.asarray(raw_hr.resize((low_w, low_h), resample=PIL.Image.BICUBIC), 'f')
                    mask = ld_mask_func(path, landmark_dict)
                                    
                    # If training => do random flip
                    if fliplr and np.random.random() < 0.5:
                        img_hr, img_lr, mask = \
                                np.fliplr(img_hr), np.fliplr(img_lr), np.fliplr(mask)
                    ## Append the images
                    hr_imgs.append(img_hr) ; lr_imgs.append(img_lr) ;  msk_imgs.append(mask)
      
                preproc = lambda x: np.array(x) / 127.5 - 1    
                hr_imgs, lr_imgs, msk_imgs = \
                         preproc(hr_imgs), preproc(lr_imgs), np.array(msk_imgs)
               
                #yield hr_imgs, lr_imgs, msk_imgs
                yield lr_imgs, hr_imgs
                ## TODO : Refactor - split the with_mask and without mask by yield from.
        
        all_img_path = glob(self.data_src_path+'/*'+ext)
        
        shuffled and np.random.shuffle(all_img_path) ## already randomization the path
        rnd_path = all_img_path
        
        mask_ld_dict = {'CASIA_lab':get_iris_mask, 'CelebA':get_face_mask}
        ## Packtor
        ld_datagen = get_batch_img(rnd_path, mask_ld_dict)
        return ld_datagen
    
    
    ##  For loading the demo data, 
    ##      which will not assert the "no-duplicate" of loading the image.
    ##      In the demo, fliplr is turn on by default!
    def ld_demo_data(self, batch_size=1, fliplr=True, ext='bmp'):
        print("path -- ", join(self.data_src_path, "*.{}".format(ext)))
        path = glob(join(self.data_src_path, "*.{}".format(ext)))
        batch_imgs_path = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        h, w = self.hr_img_size
        low_h, low_w = (h // self.scalr), (w // self.scalr)

        for idx, img_path in enumerate(batch_imgs_path):
            raw_hr, img_hr = PIL.Image.open(img_path), np.asarray(PIL.Image.open(img_path))
            img_lr = np.asarray(raw_hr.resize((low_w, low_h), resample=PIL.Image.BICUBIC))
            
            if fliplr and np.random.random() < 0.5:
                img_hr, img_lr = np.fliplr(img_hr), np.fliplr(img_lr)
                
            imgs_hr.append(img_hr) ; imgs_lr.append(img_lr)
            
        preproc = lambda x: np.array(x) / 127.5 - 1
        imgs_hr, imgs_lr = preproc(imgs_hr), preproc(imgs_lr)
        
        return imgs_lr, imgs_hr
    
    
    ## TODO : Dangling parameter - batch_size
    def get_img_generator(self, batch_size=1, given_dataset_name=None, file_ext=None):
        
        ## generator of loading all image in dataset.
        def get_batch(imgs_path):
            h, w = self.hr_img_size
            low_h, low_w = (h // self.scalr), (w // self.scalr)
            
            for path in imgs_path: 
                raw_hr = PIL.Image.open(path).convert("RGB")
                # raw_hr.mode
                img_lr = np.asarray(raw_hr.resize((low_w, low_h), resample=PIL.Image.BICUBIC))
                bt_img_lr = np.array([img_lr]) / 127.5 -1          # norm to [-1, 1] range
                file_name = path.split(os.sep)[-1].split('.')[-2]  # get file name 
                yield bt_img_lr, file_name
                
        assert(file_ext is not None), "ERROR_MESSAGE : The given file extension is None.."
        imgs_path = glob(join(self.data_src_path, "*.{}".format(file_ext))) \
            if given_dataset_name is None \
                else glob("../datasets/{}/*.{}".format(given_dataset_name, file_ext))
        
        imgs_generator = get_batch(imgs_path)
        return imgs_generator

if __name__ == "__main__":
    pass
    