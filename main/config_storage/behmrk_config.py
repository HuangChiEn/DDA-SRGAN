#!/usr/bin/env python3
# -*- coding: utf-8 -*-   ## In the case to print chinese character.
import argparse

# config file for the other SR-method 
def get_config():
    ## (1) Parameter setting stage : the parameters used in module are defined by following code : 
    parser = argparse.ArgumentParser(description="This config for benchmark, you don't need much setting")
    
    ## Execution enviroment setting : 
    parser.add_argument('--module_name', type=str, default="RCAN", help="model name of SR benchmark ")
    parser.add_argument('--lr_height', type=int, default=120)
    parser.add_argument('--lr_width', type=int, default=160)
    parser.add_argument('--cuda', type=str, default="0, 1, 2, 3", help='a list of gpus.')
    parser.add_argument('--force_cpu', type=bool, default=False, help='Force cpu execute the assignments of model whatever gpu assistance.')
    
    args = parser.parse_args() # parser the arguments, get parameter via arg.parameter

    return args
