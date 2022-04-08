#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:34:57 2020

@author: joseph
"""
## TODO.0 : add the hash_tag to preserve the new weight file.
## TODO.1 : modify the save_model_weight to the same style as load_model_weight
from os.path import join, isdir
import numpy as np

## while ~ Obviously the code is un-complete, but i don't want to deal with these junk code, until i have a lot of time to refactor(rebuild) it....zzz
class ModMang:  ## _log_regist : building..
    
    def __log_regist(self):
        with open(join(self.save_direc, "record_of_weight")) as file:
            file.write()  ## ?? what ! log_register is ?
            
        
    def __init__(self, save_direc=None, mod_record=None):
        try:
            if save_direc is None:
                raise ValueError("The given directory is None, please offer a directory.")
            elif isdir(save_direc) is False:
                raise ValueError("The given directory is not exist, please check the path.")
            else:
                ## should build the save directory via json setting file.
                self.save_direc = save_direc
        except ValueError as ex:
            print("ERROR_MESSAGE : ", ex)
            
        #self.mod_record['']   ## unremove commented code snippet ..=.=/// oh
        
    ''' Save RRDB model pretrain model : '''
    def save_RRDB(self, RRDB=None, mod_log=None):  
        ## For the model include Lambda Layer in keras, whose architecture can not be saved. 
        ##     Saving the weight is the only way to preserve the model.
        assert (RRDB is not None), ("ERROR_MESSAGE : RRDB is not given.")
        
        try:
            RRDB.save_weights(self.save_direc + '/RRDRB/RRDRB_weight.h5')
        except Exception as ex :
            print(ex)
        
    def save_model_weight(self, generator=None, discriminator=None, mod_log=None):
        assert (generator is not None) or (discriminator is not None), \
                ("ERROR_MESSAGE : G and D are not given.")
        try:
            if (generator is not None) and (discriminator is not None):
                generator.save_weights(self.save_direc + '/generator/test.h5')
                discriminator.save_weights(self.save_direc + '/discriminator/Rela_Dis_weight.h5')
                #self.__log_regist(mod_log, )   ## not good ~ ~ 
            elif generator is not None:
                generator.save_weights(self.save_direc + '/generator/tst_1.h5')
                
            else:
                discriminator.save_weights(self.save_direc + '/discriminator/Rela_Dis_weight.h5')
                
        except Exception as ex:
            print(ex)
            
            
    ##  Very mass  and unflexible..
    def load_model_weight(self, RRDB=None, generator=None, discriminator=None, msk_generator=None):
        ## Of course the model can not all be None..
        assert ((generator is not None) or (discriminator is not None) or (RRDB is not None) or (msk_generator is not None)), \
                "ERROR_MESSAGE : The given model instance are all be None."
        
        try:
            if generator is not None and discriminator is not None:
                generator.load_weights(self.save_direc + '/generator/test.h5')
                discriminator.load_weights(self.save_direc + '/discriminator/Rela_Dis_weight.h5')
                print('loading weight of models success, procedure continue..')
                return generator, discriminator
            
            elif generator is not None:
                generator.load_weights(self.save_direc + '/generator/tst_1.h5') 
                print('loading weight of model success, procedure continue..')
                return generator
            
            elif msk_generator is not None:
                msk_generator.load_weights(self.save_direc + '/att_dis/CA_exp001_0.h5')
                print('loading weight of model success, procedure continue..')
                return msk_generator
            else:
                RRDB.load_weights(self.save_direc + '/RRDB/RRDRB_weight.h5') 
                print('loading weight of model success, procedure continue..')
                return RRDB
            
        except Exception as inst:   
            print("load model failure : " + str(inst))
    
if __name__ == "__main__":
    print('no unitest\n')
    pass