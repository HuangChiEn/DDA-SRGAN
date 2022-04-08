# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:06:49 2021
Description : The Data_Bedrock provides an light-weight framework for control 
                the data-flow in dataloader. 

@author: josep
"""
from pathlib import Path
import numpy as np
import random
from functools import partial


class Data_Stone:
    # HACKME ~~ support more data type, haha.
    support_type = ['dir']  # 'txt', 'csv', 'mat', 'json'
    
    def __init__(self, data_path=None, ext=None, cmp_func=None, reader=None, reader_args={}):
        self.path_obj = Path(data_path)
        assert self.path_obj.exists()
        self.src_typ = 'dir' if self.path_obj.is_dir() else None
        
        if self.src_typ is None:
            self.src_typ = self.path_obj.name.split('.')[-1]
        
        assert self.src_typ in Data_Stone.support_type, "The target type {} is not supported \n supported format : {}\n".format( \
            self.src_typ, Data_Stone.support_type)
        
        self.ext = ext
        self.cmp_func = cmp_func
        
        # build the data source
        src_dict = {'dir' : self.__flow_from_dir}
        src_dict[self.src_typ]()
        
        # build the default reader
        def_reader = lambda x : x
        self.__Reader = def_reader if reader is None else reader
        # froze the args used in reader
        self.__Reader = partial(self.__Reader, **reader_args)
        
    @property
    def reader(self):
        return self.__Reader
    
    @reader.setter
    def reader(self, custom_reader):
        self.__Reader = custom_reader
        
    @property
    def data_src(self):
        return self.__Data_src
    
    def __flow_from_dir(self):
        target = '*' if self.ext is None else '*.' + self.ext 
        self.__Data_src = list(self.path_obj.glob(target))
        if self.cmp_func is not None:
            self.__Data_src.sort(key = lambda path_obj : self.cmp_func(str(path_obj)))
        
     
class Data_Packtor:
    @staticmethod
    def pkg_src2gen(source=None, batch_size=None, shuffle=True):
        
        def shuf_fun(seed, src_lst):
            [random.Random(seed).shuffle(src) for src in src_lst]
            
        assert source is not None
        pass    
        
    
    @staticmethod
    def custom_pkg_gen(stn_lst=None, batch_size=1, shuffle=True, 
                       pkg_func=None, pkg_args={}):
        def chk_src():
            assert stn_lst is not None
            assert pkg_func is not None
            
        def shuf_fun(seed, src_lst):
            [random.Random(seed).shuffle(src) for src in src_lst]
            
        chk_src()
        src_lst = [stn.data_src for stn in stn_lst]
        readers = [stn.reader for stn in stn_lst]
        
        iter_siz = len(src_lst[0]) // batch_size
        
        while True: 
            # shuffle & combine the source into zip iter
            shuffle and shuf_fun(42, src_lst)
            iter_obj = zip(*src_lst)
            
            for _ in range(iter_siz):
                tmp_lst = []
                
                for _ in range(batch_size):
                    src_items = next(iter_obj)
                    data = [ readers[idx](str(src)) for idx, src in enumerate(src_items) ]
                    
                    # Ragged shape object can not directly convert into np.array
                    template = np.empty(len(data), dtype=object)
                    template[:] = data
                    tmp_lst.append(template)
                
                data_mtr = np.array(tmp_lst)
                siz = data_mtr.shape[1]
                all_data = []
                for idx in range(siz):
                    all_data.append(data_mtr[:, idx])
                    
                yield pkg_func(*all_data, **pkg_args)


if __name__ == "__main__":
    pass
    
    