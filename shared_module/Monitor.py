#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:29:07 2021

@author: joseph
"""

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
   
class Monitor(object):
    
    def __init__(self, pretrain_dir, model_type, dataset_name, exp_tag=None):
        save_folder = join(pretrain_dir, model_type, dataset_name)
        assert(isdir(save_folder)), "The given path do not contain the folder.."
        self.save_folder = save_folder
        key_seq = ['model_type', 'loss_weight', 'learning_rate', 'optimizer', 
                   'epochs', 'batch_size']
        self.model_config = dict.fromkeys(key_seq)
        
        self.exp_tag = exp_tag
        
    ## Write log file may be deprecated for place into monitor module in the furture
    def write_log_file(module, log_name):
        log_file_path = join(self.save_folder, module, "log_file", self.exp_tag)
        with open(log_file_path, "w+") as f_ptr:
            file_lst = ["Training Parameters : \n"]
            file_lst.extend([ "{0}. {1} ->  {2}\n".format(idx, key, value) 
            for idx, (key, value) in enumerate(self.model_config.items())])
                f_ptr.writelines(file_lst)
    
    @staticmethod
    def write_module_png(module, save_path):
        assert(save_path.split(".")[-1] == "png"),  \
                "the module will be written into png files"
        dot_obj = model_to_dot(module)
        dot_obj.write_png(save_path)
        
        
if __name__ == "__main__":
    pass