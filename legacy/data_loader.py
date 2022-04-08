# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:52:29 2020

#TODO list :

# Change_log : 

@author: josep
"""

## Read aid information
import scipy.io as sio

## Image I/O
import cv2

## For glob the path
import glob
 
## other utils
from os.path import join, isfile, isdir, exists, splitext
import numpy as np
from collections import OrderedDict

class Data_Loader():
    
    def __init__(self, src_dict={}, fit_mode=None, def_ld_tuple=()):
        '''
        @Input : 
            src_dict - a dictionary with two element {'data_src', 'label_src'},
                        data_src - key, the value is the path of data source.
                        label_src - key, the value is the path of label source.
                        
            def_ld_tuple - a tuple with two element (src_ext, ld_func),
                            src_ext - string, extension of data source.
                            ld_func - function, self define for loading data.
        '''
        
        def chk_src():
            assert("data_src" in src_dict), "data_src must be given."
            assert(exists(src_dict["data_src"])), "The given data source is not exists."
            self.data_src = src_dict["data_src"]
            
            if ("label_src" in src_dict):    
                assert(exists(src_dict["label_src"])), \
                        "Note : The given label source is not exists.\n"
                self.lab_src = src_dict["label_src"]
            else:
                print("Note : label_src will not given, switch to prediction mode.\n")
                self.lab_src = None
                
        ## check whatever the type extension is match.
        def chk_src_typ(chk_func, src_ext):
            def path_wrap(path):
                if chk_func(path) and splitext(path)[-1] == src_ext:
                    return True
                return False
            return path_wrap
        
        ## define how to load the source of data and label.
        def flow_from_txt(src):
            with open(src) as f_ptr:
                path_lst = [lin.strip() for lin in f_ptr]
            return path_lst
    
        def flow_from_dir(src):
            return glob.glob(join(src, "*"))
            
        ## new attribute..
        def flow_from_mat(src, key):
            mat_cnt = sio.loadmat(src)
            return mat_cnt[key]


        ## check the source, label is optional.
        chk_src()
        
        ## Set the check function as the key of corresponding load function.
        ld_func_dict = OrderedDict([ 
                          ( chk_src_typ(isfile, ".txt"), flow_from_txt ),
                          ( chk_src_typ(isfile, ".mat"), flow_from_mat ),
                          ( isdir, flow_from_dir )                  ])
        
        if def_ld_tuple:
            ## self define loader will be place into the first index (index 0).
            chk_func_trf = lambda src_ext: chk_src_typ(isfile, src_ext)
            def_ld_tuple[0] = chk_func_trf(def_ld_tuple[0])
            ld_func_dict = OrderedDict( [def_ld_tuple] + ld_func_dict.items() )
        
        ## Data source and Label source corresponding load function.
        get_ld_func = lambda x : [ ld_func for chk, ld_func in ld_func_dict.items() if chk(x) ]
        data_ld_lst, lab_ld_lst = get_ld_func(self.data_src), get_ld_func(self.lab_src)
        
        self.data_path_lst = np.array(data_ld_lst[0](self.data_src)) if (data_ld_lst) else []
        self.lab_path_lst = np.array(lab_ld_lst[0](self.lab_src)) if (lab_ld_lst) else []
        
    
    ## Private function -
    def __data_gen(self, wrapper):
        path_gen = (path for path in self.data_path_lst)
        
        ## Default transformer is identity function.
        transformer = self.data_transformer if self.data_transformer is not None \
                    else (lambda x : x)
                    
        while True:
            img_lst = [transformer(cv2.imread(next(path_gen))) \
                               for _ in range(self.batch_size)]
            yield wrapper(img_lst)
            
        
    def __label_gen(self, wrapper):
        path_gen = (path for path in self.lab_path_lst)
        
        ## Default encoder is identity function.
        encoder = self.label_encoder if self.label_encoder is not None \
                    else (lambda x : x)
        
        while True:
            img_lst = [encoder(next(path_gen)) \
                               for _ in range(self.batch_size)]
            yield wrapper(img_lst)
                
    
    def get_generator(self, batch_size=1, shuffle=True, data_wrapper=np.array, lab_wrapper=np.array, 
                            label_encoder=None, data_transformer=None):
        
        def shuf_data(data_size):    
            shuf_idx = np.arange(data_size)
            np.random.shuffle(shuf_idx)
            self.data_path_lst[:] = self.data_path_lst[shuf_idx[:]]
            self.lab_path_lst[:] = self.lab_path_lst[shuf_idx[:]]
            
        self.batch_size = batch_size
        self.label_encoder = label_encoder
        self.data_transformer = data_transformer
        data_size = len(self.data_path_lst)
        iteration = int(data_size//batch_size)
        
        if shuffle:
            shuf_data(data_size)
        
        for _ in range(iteration):
            yield from zip(self.__data_gen(data_wrapper), self.__label_gen(lab_wrapper))
            

if __name__ == "__main__":
    ## composer.. 
    def composer(instance):
        def resiz_wrp():
            gray_img = cv2.cvtColor(instance, cv2.COLOR_RGB2GRAY)
            return cv2.resize(gray_img, (200, 300))
            
        return np.array(resiz_wrp()) 
    
    
    src_dict = {"data_src":"HR", "label_src":"boundaries"}
    data_ld = Data_Loader(src_dict=src_dict)
    data_gen = data_ld.get_generator(batch_size=2, data_transformer=composer)
    
    for _ in range(10):
        a, b = next(data_gen)
        print(b)
        cv2.imshow("img", a[0])
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    