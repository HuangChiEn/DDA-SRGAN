#!/usr/bin/env python3
# -*- coding: utf-8 -*-   ## In the case to print chinese character.
import glob 
import importlib
import os

def load_config(module_name=None, exp_tag='000'):
    module_config = importlib.import_module("." + module_name, package="config_storage")
    return module_config.get_config()

if __name__ == '__main__':
    pass
    
