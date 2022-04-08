#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:46:23 2020

@ Decsription : The model manager that is, a server for loading & saving the model status, 
                which including model weight, model related information(learning rate, loss weights).
                
                The manager will help you automatically write the infomation of model into the logfile,
                which under the pretrain folder.
                
                have fun www.. as your childhood, and more creative to the tedious work ~ ~ haha
@author: No_Name
"""

from os.path import isdir, join
from os import makedirs

class modelManager:
    
    def __init__(self, pretrain_dir, model_type, dataset_name, exp_tag=None):
        save_folder = join(pretrain_dir, model_type, dataset_name)
        assert(isdir(save_folder)), "The given path do not contain the folder.."
        self.save_folder = save_folder
        key_seq = ['model_type', 'loss_weight', 'learning_rate', 'optimizer', 
                   'epochs', 'batch_size']
        self.model_config = dict.fromkeys(key_seq)
        
        self.exp_tag = exp_tag

        
    def save_model_weight(self, generator=None, discriminator=None, model_config_dict=None):
        
        ##Write log file may be deprecated for place into monitor module in the furture.
        def write_log_file(module, log_name):
            log_file_path = join(self.save_folder, module, "log_file", self.exp_tag)
            with open(log_file_path, "w+") as f_ptr:
                file_lst = ["Training Parameters : \n"]
                file_lst.extend([ "{0}. {1} ->  {2}\n".format(idx, key, value) 
                                    for idx, (key, value) in 
                                        enumerate(self.model_config.items())])
                f_ptr.writelines(file_lst)
                
        if model_config_dict is not None:
            ## Addition info will be appended to the tail of dict
            for key, value in model_config_dict.items():
                self.model_config[key] = value
                
        try:  
            if generator is not None:
                module = "generator"
                gen_save_path = join(self.save_folder, module, (self.exp_tag+".h5"))
                print("Saving the {} into following path :\n {}".format(module, gen_save_path))
                generator.save_weights(gen_save_path)
                write_log_file(module=module, log_name=(self.exp_tag+".txt"))
                
            if discriminator is not None:
                module = "discriminator"
                dis_save_path = join(self.save_folder, module, (self.exp_tag+".h5"))
                print("Saving the {} into following path :\n {}".format(module, dis_save_path))
                discriminator.save_weights(dis_save_path)
                write_log_file(module="discriminator", log_name=(self.exp_tag+".txt"))
                
        except Exception as ex:
            print("save model failure : " + str(ex))
            
            
    def load_model_weight(self, generator=None, discriminator=None):
        
        try:
            if generator is not None:
                module = "generator"
                gen_ld_path = join(self.save_folder, module, (self.exp_tag+".h5"))
                generator.load_weights(gen_ld_path) 
                print('loading weight of model success from {}\n'.format(gen_ld_path))
                print('Procedure continue....')
            
            if discriminator is not None:
                module = "discriminator"
                dis_ld_path = join(self.save_folder, module, (self.exp_tag+".h5"))
                discriminator.load_weights(dis_ld_path)
                print('loading weight of model success from {}\n'.format(dis_ld_path))
                print('Procedure continue....')
            
        except Exception as ex:   
            print("load model failure : " + str(ex))
            
        return generator, discriminator

    ## part of functionality for Model_check_point in keras
    
    def save_ckpt(self, generator=None, discriminator=None, 
                  gen_loss_val=None, dis_loss_val=None):
        ## makedirs
        try:  
            if generator is not None:
                module = "generator"
                ckpt_path = self.exp_tag + "_" + str(gen_loss_val) +".ckpt" 
                gen_save_path = join(self.save_folder, "generator", "ckpt", ckpt_path)
                print("Saving the {} into following path :\n {}".format(module, gen_save_path))
                generator.save_weights(gen_save_path)
                
            if discriminator is not None:
                module = "discriminator"
                ckpt_path = self.exp_tag + "_" + str(dis_loss_val) +".ckpt" 
                dis_save_path = join(self.save_folder, "discriminator", "ckpt", ckpt_path)
                print("Saving the {} into following path :\n {}".format(module, dis_save_path))
                discriminator.save_weights(dis_save_path)
                
        except Exception as ex:
            print("save model failure : " + str(ex))
        
        
if __name__ == "__main__":
    pass
