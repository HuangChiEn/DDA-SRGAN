#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:03:21 2021


Pytorch like dataloader : 

@author: joseph
"""

from Data_Bedrock import Data_Stone, Data_Packtor

from functools import reduce, partial
import numpy as np
import re
import os
from os.path import join

import imageio
import cv2


class Data_loader:
    # Utility function
    # For the order alignment of the input lr and the corresponding mask
    # (cls1_1.lr, cls1_1.msk, cls1_2.lr, cls1_2.msk, ...)
    def __cmp_func(self, path):
        ## this part is disable ~~
        def sort_CASIA_lab():
            def dig2num(digs):
                digs.reverse()
                sum_lst = [int(dig)*(10**idx) for (idx, dig) in enumerate(digs, 0)]
                return reduce(lambda val, elem : val + elem, sum_lst, 0)
            # /root/path/file_dir/cls_12_1.bmp -> os.sep == "/", 
            file_name_lst = path.split(os.sep)[-1].split('_')
            # get class (e.g. 12)
            dig_1 = re.findall(r'\d', file_name_lst[0])
            # get member (e.g. 1)
            dig_2 = re.findall(r'\d', file_name_lst[1])
            
            return dig2num(dig_1) * 10 + dig2num(dig_2)
        
        # CelebA have well-define filename, just use the default sort
        return sort_CASIA_lab
        
        
    # HACKME : the mask may not necessary place into parent folder
    def __get_iris_mask(self, path):
        path_lst = path.split(os.sep)
        ## Add the mask label in the filename
        file_name_lst = path_lst[-1].split('.')
        mask_name = file_name_lst[-2] + "_mask." + file_name_lst[-1]
        ## joint back to the path from lst.
        mask_dir = os.sep.join(path_lst[:-1])
        mask_path = join(mask_dir, "msk", mask_name)
        return imageio.imread(mask_path)
    
    def __get_face_mask(self, path, msk_shap, landmrk_loc):
        
        def load_landmark():
            ## Generate the face landmark : 
            landmrk_dict = {}
            
            with open(landmrk_loc) as f_ptr:
                f_ptr.readline() ; f_ptr.readline() ## get rid of reduandant info.
                
                for lin in f_ptr:
                    cords_lst = lin.strip().split()
                    filNam = cords_lst.pop(0)
                    # Trace by README.md in CelebA dataset
                    ## p1 : (lefteye_x lefteye_y), p2 : (righteye_x righteye_y),
                    ## p3 : (nose_x nose_y), p4 : (leftmouth_x leftmouth_y),
                    ## p5 : (rightmouth_x rightmouth_y)
                    lftEye = (cords_lst[0], cords_lst[1]) ; rgtEye = (cords_lst[2], cords_lst[3])
                    lftMou = (cords_lst[6], cords_lst[7]) ; rgtMou = (cords_lst[8], cords_lst[9])
                    nose = (cords_lst[4], cords_lst[5])
                    landmrk_dict[filNam] = [lftEye, rgtEye, lftMou, rgtMou, nose]
                    
            return landmrk_dict
        
        
        # HACKME : make the ROI region according to the ratio of the img size.
        # in short -> dynamic margin
        # Prior margin for CelebA : 
        #   (w:40, h:60) -> (w/4, h/6) -> margin : ( 10, 10 )
        
        # Draw the white point(pnt) facial landmark with the margin,
        #      on the black map.
        def cords2msk(cords, margin=15):
            # initial black mask
            landmrk_msk = np.zeros(msk_shap)
            
            for pnt in cords:
                pnt = [int(pnt[0]), int(pnt[1])]
                pnt = np.array(pnt)
                up_pnt = pnt + margin
                dwn_pnt = pnt - margin 
                # fill the ROI region (with value 1) according to the 5 facial point
                landmrk_msk[dwn_pnt[0]:up_pnt[0]+1, dwn_pnt[1]:up_pnt[1]+1] = 1.0
                
            return landmrk_msk 
        
        # Record all coordinate of landmarks, and mapping it by given key
        landmrk_dict = load_landmark()
        path_lst = path.split(os.sep)
        file_name = path_lst.pop(-1)
            
        cords = landmrk_dict[file_name]
        landmrk_msk = cords2msk(cords)
            
        return landmrk_msk
        
    
    def __init__(self, data_set_path=None, hr_img_size=None, scalr=None, ext=None):
        # Image source prepare
        self.data_set_path = data_set_path
        self.hr_shp = hr_img_size
        self.ext = ext
        self.data_stn = Data_Stone(data_set_path, ext=self.ext, reader=imageio.imread)
        self.scalr = scalr
        # Mask source preprae
        self.msk_ld_dict = {'CASIA_lab':self.__get_iris_mask, 'CelebA':self.__get_face_mask}
        self.msk_shape = hr_img_size
    
        
    def ld_data_gen(self, batch_size=1, fliplr=False, shuffle=True, 
                    **_):
        
        def pkg_func(batch_img, lr_shp=(160, 120), hr_shp=(640, 480), 
                     dwn_samp=None, hr_samp=None):
            norm = lambda x: np.asarray(x) / 127.5 - 1  
            
            lr_imgs = [dwn_samp(img) for img in batch_img]
            hr_imgs = [hr_samp(img) for img in batch_img]
            
            lr_imgs, hr_imgs = norm(lr_imgs), norm(hr_imgs)
            return lr_imgs, hr_imgs
            
        [h, w] = self.hr_shp
        lr_shp = (w // self.scalr), (h // self.scalr)
        dwn_samp = lambda x : cv2.resize(x, lr_shp, interpolation=cv2.INTER_CUBIC)
        hr_samp = lambda x : cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
        
        args = {'dwn_samp':dwn_samp, 'hr_samp':hr_samp}
        
        return Data_Packtor.custom_pkg_gen([self.data_stn], batch_size=batch_size, 
                                           pkg_func=pkg_func, pkg_args=args)
        
    
        
    def ld_msk_data_gen(self, batch_size=1, fliplr=False, shuffle=True, 
                    msk_typ='CASIA_lab', msk_path=None,
                    **_):
   
        def pkg_func(batch_img, msk_src, dwn_samp, hr_samp):
            norm = lambda x: np.asarray(x) / 127.5 - 1  
            unroll_shape  = lambda  x: np.asarray(list(x))
            
            lr_imgs = [dwn_samp(img) for img in batch_img]
            hr_imgs = [hr_samp(img) for img in batch_img]
            
            
            lr_imgs, hr_imgs, msk_src = \
                    norm(lr_imgs), norm(hr_imgs), unroll_shape(msk_src)
            
            return lr_imgs, hr_imgs, msk_src
        
        
        assert msk_typ in self.msk_ld_dict.keys(), \
            "No such mask type {}, supported mask type : {}\n".format(msk_typ, 
                                                                    self.msk_ld_dict.keys())
        # Hard code the location of the CelebA Annotation list  
        dict_path = join("..", "datasets", "CelebA", "Anno", "list_landmarks_align_celeba.txt")
        args = {'landmrk_loc':dict_path, 'msk_shap':self.hr_shp} if msk_typ == 'CelebA' else {}
        msk_stn = Data_Stone(msk_path, reader=self.msk_ld_dict[msk_typ],
                                  reader_args=args, ext=self.ext)
        
        [h, w] = self.hr_shp
        lr_shp = (w // self.scalr), (h // self.scalr)
        dwn_samp = lambda x : cv2.resize(x, lr_shp, interpolation=cv2.INTER_CUBIC)
        hr_samp = lambda x : cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
        
        args = {'dwn_samp':dwn_samp, 'hr_samp':hr_samp}
        
        return Data_Packtor.custom_pkg_gen([self.data_stn, msk_stn], 
                                               batch_size=batch_size, 
                                                   pkg_func=pkg_func, pkg_args=args)
        

    def ld_generate_data_gen(self, batch_size=1, fliplr=False, shuffle=True, 
                    **_):                 #  160, 120 fill into lr_shp~ don't forget!!
        def pkg_func(batch_img, file_path, lr_shp=(40, 50), dwn_samp=None):
            norm = lambda x: np.asarray(x) / 127.5 - 1  
        
            lr_imgs = [dwn_samp(img) for img in batch_img]
            lr_imgs = norm(lr_imgs)
            file_names = [ path.split(os.sep)[-1] for path in file_path]
            name_wo_ext = np.asarray([ file_name.split(".")[-2] for file_name in file_names] ).squeeze()
            
            return lr_imgs, name_wo_ext
        
        file_name_stn = Data_Stone(self.data_set_path, ext=self.ext)
        
        [h, w] = self.hr_shp
        lr_shp = (w // self.scalr), (h // self.scalr)
        dwn_samp = lambda x : cv2.resize(x, lr_shp, interpolation=cv2.INTER_CUBIC)
        
        wrap_pkg_func = partial(pkg_func, dwn_samp=dwn_samp)
        return Data_Packtor.custom_pkg_gen([self.data_stn, file_name_stn], batch_size=batch_size, 
                                           pkg_func=wrap_pkg_func)
        
        