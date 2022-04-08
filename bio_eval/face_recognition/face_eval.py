#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:23:55 2020

@author: joseph
"""

import numpy as np
from os.path import join, basename
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import pickle

## kernel face module
#import face_recognition as fac_recg
from tqdm import tqdm 
from Facenet_recognizer.recognizer import Face_recognizer as Face_Recog
from PIL import Image, ImageDraw
import scipy.io as sio
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.spatial import distance
import cv2




## Calculate face recognition performance and plot the 'static' roc curve
def recog_eval(conf_path_lst, lab_path_lst, tag_lst, save_path="roc.png", prefix="./"):
    
    def plot_curve(metric_dict, tag_lst):
        fig = plt.figure()
        for (roc_auc, (fpr, tpr)), tag in zip(metric_dict.items(), tag_lst):
            plt.plot(fpr, tpr, lw=1, label='tag {} ; (area={})'.format(tag, roc_auc))
            
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Baseline of random')
        plt.xlim([0.0, 1.0]) ; plt.ylim([0.0, 1.05])
        plt.xlabel('FPR') ; plt.ylabel('TPR')
        plt.legend(loc='lower right')
        plt.show()
        fig.savefig(save_path)
    
    def compute_eer(fpr, tpr, threshold):
        fnr = 1 - tpr
        EER_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        
        return EER, EER_threshold 
    
    def get_imp_auth(label_vec, conf_vec):
        imp, auth = [], []
        for lab, dist in zip(label_vec, conf_vec):
            if lab == 1:
                auth.append(dist)
            else:
                imp.append(dist)
        return imp, auth
    
    conf_path_lst, lab_path_lst = [ prefix+path for path in conf_path_lst ], [ prefix+path for path in lab_path_lst ]
    
    ## Load the pickle file of confusion matrix and the corresponding matrix.
    metric_dict = {}
    for conf_path, lab_path, tag in zip(conf_path_lst, lab_path_lst, tag_lst):
        with open(conf_path, "rb") as conf_ptr, open(lab_path, "rb") as lab_ptr:
            conf_tmp_mtr, lab_tmp_mtr = pickle.load(conf_ptr), pickle.load(lab_ptr)
            
        ## Normalization -      
        norm = MinMaxScaler() # into range [ 0 ~ 1 ] 
        conf_tmp_mtr = norm.fit_transform(conf_tmp_mtr)
        conf_vec, label_vec = \
                    conf_tmp_mtr.reshape(-1), lab_tmp_mtr.reshape(-1)
        
        
        ## Compute ROC, AUC, fisher_ratio 
        print("roc-curve compute phase : \n")
        fpr, tpr, threshold = roc_curve(label_vec, conf_vec)
        roc_auc = auc(fpr, tpr)
        
        metric_dict[roc_auc] = (fpr, tpr)
        print("get imposter authentication : \n")
        imp, auth = get_imp_auth(label_vec, conf_vec)
        
        fisher_ratio = (abs(np.mean(auth)-np.mean(imp)))/(np.std(auth)+np.std(imp))
        
        print("Show [ {} ] EER result\n".format(tag))
        EER, _ = compute_eer(fpr, tpr, threshold)
        print("EER : {}\n".format(EER))
        print("FPR : {} ; TPR : {}\n".format(fpr, tpr))
        print("AUC : {}\n".format(roc_auc))
        print("Fisher Ratio : {}\n".format(fisher_ratio))
        print("---------------------------------\n")
        tag = conf_path.split(".")[0] + ".mat"
        
        # saving the mat file 
        print(tag)
        sio.savemat(tag, {"fpr":fpr, "tpr":tpr, "roc_auc":roc_auc, "EER":EER})
        
    plot_curve(metric_dict, tag_lst)
    
    

def face_recog_procedure(debug_flg=False, msk_dict=None, fac_recg=None):
    
    def show_debug_info(lab, sim, reg_img, prb_img):
        print("label : {}".format(lab))
        print("Sim : {}".format(sim))
        cv2.imshow("reg_img", reg_img)
        cv2.imshow("prb_img", prb_img)
        cv2.waitKey(0)
        cv2.destroyWindow("reg_img")
        cv2.destroyWindow("prb_img")
    
    ## Face Embedding calculation : 
    def calc_embs(cls_lab, filepaths, fix=True):
        cls_lab = np.array([ ([lab]*10) for lab in cls_lab ]) # each class have 10 image is what we already known. 
        cls_lab = cls_lab.reshape(-1)  # side-effect of list comprehension.
        embd_feas = []
        
        with tqdm(total=len(filepaths)) as pbar:
            for lab, path in zip(cls_lab , filepaths):
                # ugly, should delete
                filename = basename(path)
                    
                if fix:
                    tmp = path.split("/")
                    name = tmp[-1].split(".")[0]
                    tmp[-1] = name+".png"
                    path = "/".join(tmp)
                
                img = cv2.imread(path)
                face_fea = fac_recg.face_encodings(img, msk_dict[filename])
                embd_feas.append( (lab, face_fea[0], img) ) 
                pbar.update()
        
        return embd_feas
    

    ## Read the file list -
    cls_lab, reg_set, prb_set = [], [], []
    
    with open("dataset/reg_set.txt", "r") as reg_ptr, \
         open("dataset/prb_set.txt", "r") as prb_ptr:
        for idx, (reg_lin, prb_lin) in enumerate(zip(reg_ptr, prb_ptr)):
                if idx % 2 == 0:
                    cls_lab.append(reg_lin.strip())
                else:
                    tmpLst_1, tmpLst_2 = reg_lin.split(), prb_lin.split()
                    reg_set.extend(tmpLst_1)
                    prb_set.extend(tmpLst_2)
        
    ## All data place into eval_set
    reg_set = [join('..', '..', 'datasets', 'CelebA', 'eval_set', path) \
               for path in reg_set]
    ## Prob set for generated data
    
    prb_set = [join('dataset', 'gen_data', 'SWCA_GAN', 'exp000_CelebA', path) \
               for path in prb_set]
    
    #prb_set = [join('dataset', 'gen_data', 'behmrk', 'RCAN', 'exp000_CelebA', path) \
    #           for path in prb_set]
    
    ## Prob set for GT data
   # prb_set = [join('..', '..', 'datasets', 'CelebA', 'eval_set', path) \
     #          for path in prb_set]
    
    
    ## Embedding projection -
    print(" prbset feature calculation..")
    prb_emb_feas = calc_embs(cls_lab, prb_set, False)
    
    print("\n\n regset feature calculation..")
    reg_emb_feas = calc_embs(cls_lab, reg_set, False)
    
    
    ## Eval set structure + 
    ##     class_name - 20 images filename list
    siz = len(reg_emb_feas)
    conf_mtr = np.zeros([siz, siz])
    label_mtr = np.zeros([siz, siz])
    
    ## Calculate normalized distance : 
    print("distance calculation - \n")
    
    with tqdm(total=(siz*siz)) as pbar:
        for idx, (reg_lab, reg_emb, reg_img) in enumerate(reg_emb_feas):
            
            for jdx, (prb_lab, prb_emb, prb_img) in enumerate(prb_emb_feas):
                label_mtr[idx][jdx] = 0.0 if (reg_lab == prb_lab) else 1.0 
                conf_mtr[idx][jdx] = fac_recg.face_distance([reg_emb], prb_emb) 
                
                if debug_flg:
                    print("idx-{} ; jdx-{} ; label : {} ; Sim : {}".format(idx, jdx, label_mtr[idx][jdx], conf_mtr[idx][jdx]))
                    show_debug_info(label_mtr[idx][jdx], conf_mtr[idx][jdx], 
                                    reg_img, prb_img)
                    
                pbar.update()
                
            
    ## Saving the calculation result -
    print("saving result -\n")
    with open("conf_mtr_dda.pk", "wb+") as conf_ptr, open("lab_mtr_dda.pk", "wb+") as lab_ptr:
        pickle.dump(conf_mtr, conf_ptr)
        pickle.dump(label_mtr, lab_ptr)
        
        
        
def get_msk_dict():
    def s2d(s):  # I know it's ugly.. if i have time i will refactor it 
        return int(float(''.join([ c for c in s if not c == ','] )))
    
    msk_dict = dict()
    
    with open("bbox_lab.txt", "r") as f_ptr:
        f_ptr.readline()
        for lin in f_ptr:
            lin = lin.strip()
            lin_lst = lin.split()
            msk_dict[ lin_lst[0] ] = [ s2d(s) for s in lin_lst[1:-1] ]
            
    return msk_dict
        

if __name__ == "__main__":
    
    msk_dict = get_msk_dict()
    
    face_recog_procedure(False, msk_dict, Face_Recog())
    
    # Face recognition for input LR (LR is generated by nearest-interpolation of HR's GT)
    tag_lst = ["GT", "DDA", "MASR", "RCAN", "nESRGANp", "bicubic", "LR"]    
    lab_lst = ["lab_mtr_gt.pk",  "lab_mtr_dda.pk", "lab_mtr_masr.pk", "lab_mtr_rcan.pk", "lab_mtr_esrganp.pk", "lab_mtr_bicu.pk", "lab_mtr_lr.pk"]    
    mtr_lst = ["conf_mtr_gt.pk", "conf_mtr_dda.pk", "conf_mtr_masr.pk", "conf_mtr_rcan.pk", "conf_mtr_esrganp.pk", "conf_mtr_bicu.pk", "conf_mtr_lr.pk"]     
    
    recog_eval(["conf_mtr_dda.pk"], ["lab_mtr_dda.pk"], ["dda"], "face_compare.png")
    
