# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 21:19:30 2021

@author: josep
"""

from pathlib import Path
import matplotlib.pyplot  as plt
from .inception_resnet_v1 import InceptionResNetV1 

import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import Normalizer


class Face_recognizer:
    
    def __init__(self, pretrain_path=None, ld_pick=True):
        path = Path("Facenet_recognizer/weights")
        self.wei_dir_obj = path.joinpath("facenet_keras_weights.h5")
        
        self.l2norm = Normalizer(norm='l2')
        
        ## recognition model prepare 
        self.model = InceptionResNetV1(input_shape=(160, 160, 3), 
                                           weights_path=str(self.wei_dir_obj))
    
    
    def __prewhiten(self, im):
        if im.ndim == 4:  # deal with batch images
            axis = (1, 2, 3)
            size = im[0].size
        elif im.ndim == 3:
            axis = (0, 1, 2)
            size = im.size
        else:
            raise ValueError('Dimension should be 3 or 4')
        
        mean = np.mean(im, axis=axis, keepdims=True)
        std = np.std(im, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(size))
        norm_im = (im - mean) / std_adj
        return norm_im
    
               
    def face_encodings(self, raw_face_im, msk):
        # prerpocessing raw face image
        x, y, width, height = msk
        crp_im = raw_face_im[y:y+height, x:x+width]
        
        face_im = cv2.resize(crp_im, (160, 160))
        norm_face_im = self.__prewhiten(face_im)
        face_sample = np.expand_dims(norm_face_im, axis=0)
        
        # get normalized face embedding 
        face_emb_vec = self.model.predict(face_sample)
        norm_emb_vec = self.l2norm.transform(face_emb_vec)
        
        return norm_emb_vec
    
    
    @staticmethod
    def face_distance(face_emb, face_emb_cmp):
        return distance.euclidean(face_emb, face_emb_cmp)
    
   
    @staticmethod
    def load_image_file(path):
        return cv2.imread(path)
    
if __name__ == "__main__":
    import face_recognition as ff 
    
    im = cv2.imread("dataset/eval_set/010882.jpg")
    msk_a = (39, 69, 90, 112)
    inm = cv2.imread("dataset/eval_set/039555.jpg")
    msk_b = (54, 76, 76, 98)
    
    innm = cv2.imread("dataset/eval_set/087033.jpg")
    msk_c = (51, 82, 72, 90)
    outm = cv2.imread("dataset/eval_set/066047.jpg")
    msk_d = (49, 75, 85, 107)
    
    fac_rec = Face_recognizer()
    #im_e = ff.face_encodings(im, num_jitters=5, model='large')
    
    #im_e = fac_rec.face_encodings(im, msk_a)
    #inm_e = fac_rec.face_encodings(inm, msk_b)
    innm_e = ff.face_encodings(innm, num_jitters=5, model='large')
    outm_e = ff.face_encodings(outm, num_jitters=5, model='large')
    #innm_e = fac_rec.face_encodings(innm, msk_c)
    #outm_e = fac_rec.face_encodings(outm, msk_d)
    
    #im_sc = fac_rec.cal_dist(im_e, outm_e)
    im_sc = ff.face_distance(im_e, innm_e)
    
    print("dist : {}\n".format(im_sc))  # 1.17, 1.11 ; 0.68, 0.52
    
    
    