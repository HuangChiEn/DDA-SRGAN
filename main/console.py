#!/usr/bin/env python3
# -*- coding: utf-8 -*-   ## In the case to print chinese character.
## execute the following code in python3.x enviroment, part of python2.x code is not allowed.
from __future__ import absolute_import  

from os import environ, system, pardir
from os.path import join
## log_dump_level : 1-Info 2-Warning 3-Error 4-Fatal
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

## Import tensorflow backend -- GPU env setting : 
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF  

## some related utility : 
import importlib
import sys     # Grabe the path of modules
sys.path.append(join(pardir, "shared_module"))

## self-define package : 
from config_storage import load_config
from common_interface import SR_base_model


def env_setting(cuda_flag, force_cpu_flag):
    if cuda_flag is not None:
        environ['CUDA_VISIBLE_DEVICES'] = cuda_flag   ## Allow the env detect your GPUs.
        system('echo $CUDA_VISIBLE_DEVICES')
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True   # Avoid to run out all memory, allocate it depend on the requiring
        sess = tf.Session(config=config)
        KTF.set_session(sess)
    else:
        print('Training without gpu. It is recommended using at least one gpu.')
        if force_cpu_flag:
            print('Force CPU execution mode.')
            environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
            environ["CUDA_VISIBLE_DEVICES"] = '-1'
        else:
            raise Exception('sorry, but without gpu without training, since the memory is not enough.')


def get_args_and_model(config_name):
    ## Get the config variable "args" from "config_storage" via importlib.
    args = load_config(config_name) 
    
    ## automatic add the path and import the corresponding module
    sys.path.append(join(pardir, args.module_name))  
    #sys.path.append(join(pardir, "benchmark", args.module_name))
    importlib.import_module(args.module_name)
    
    ## GPU enviroment setting stage : (pass the n_gpu to the args)
    env_setting(args.cuda, args.force_cpu)
    
    ## Find the sr model from base class
    model_lst = SR_base_model.__subclasses__()
    model_idx = [idx for idx, cls in enumerate(model_lst) \
                   if cls.__module__ == args.module_name]
    model_templt = model_lst[model_idx[0]]
    
    return args, model_templt(**args.__dict__)
    
if __name__ == '__main__':
    # All you need to set in here is the config_name or it will be given by default.
    config_name = "DDA_SRGAN_config" if len(sys.argv) < 2 else sys.argv[1]
    args, deepMod = get_args_and_model(config_name)
    
    ## Assignments of model executing stage : 
    ## Proposed common method -->  "training", "generating", "evaluation"
    deepMod.training(**args.__dict__)
    #deepMod.generating(**args.__dict__)  
    