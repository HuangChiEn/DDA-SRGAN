#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:50:25 2020

@author: joseph
"""

   
    """
    *******************************************************************************
     FACE RECOGNITION DATA_LOADER : 
    *******************************************************************************
    self.info_dict_buffer = {'msk_tag':False, 'batch_imgs_path':[], 'rand_flip':[]}
        
    def load_celebA_face_mask(self, landmrk_loc="../datasets/CelebA/Anno/list_landmarks_align_celeba.txt"):
        ## Utility function : 
        def load_landmark():
            ## Generate the face landmark : 
            landmrk_dict = {}
            
            with open(landmrk_loc) as f_ptr:
                f_ptr.readline() ; f_ptr.readline() ## get rid of reduandant info.
                
                for lin in f_ptr:
                    cords_lst = lin.strip().split()
                    filNam = cords_lst.pop(0)
                    
                    ## p1 : (lefteye_x lefteye_y), p2 : (righteye_x righteye_y),
                    ## p3 : (nose_x nose_y), p4 : (leftmouth_x leftmouth_y),
                    ## p5 : (rightmouth_x rightmouth_y)
                    lftEye = (cords_lst[0], cords_lst[1]) ; rgtEye = (cords_lst[2], cords_lst[3])
                    lftMou = (cords_lst[6], cords_lst[7]) ; rgtMou = (cords_lst[8], cords_lst[9])
                    nose = (cords_lst[4], cords_lst[5])
                    landmrk_dict[filNam] = [lftEye, rgtEye, lftMou, rgtMou, nose]
                    
            return landmrk_dict
        
        def cords2msk(cords, margin=15):
            landmrk_msk = np.zeros(self.hr_img_size)
            
            for pnt in cords:
                pnt = [int(pnt[0]), int(pnt[1])]
                pnt = np.array(pnt)
                up_pnt = pnt + margin
                dwn_pnt = pnt - margin 
                landmrk_msk[dwn_pnt[0]:up_pnt[0]+1, dwn_pnt[1]:up_pnt[1]+1] = 1.0
                
            return landmrk_msk
        
        ## end of utility function 
        #######################################################################
        
        if (self.info_dict_buffer['msk_tag'] == False):
            raise Exception("The corresponding data has not be loaded, you should \
                            call load_data function first")
        
        landmrk_dict = load_landmark()
        landmrk_msk_lst = []
        rand_filp = self.info_dict_buffer['rand_flip']
        
        for idx, img_path in enumerate(self.info_dict_buffer['batch_imgs_path']):
            path_lst = img_path.split('/')
            file_name = path_lst.pop(-1)
            
            cords = landmrk_dict[file_name]
            landmrk_msk = cords2msk(cords)
            
            if rand_filp and rand_filp[idx] < 0.5:
                landmrk_msk = np.fliplr(landmrk_msk)
            
            landmrk_msk_lst.append(landmrk_msk)
        
        ## resetting the buffer.
        self.info_dict_buffer = {'msk_tag':False, 'batch_imgs_path':[], 'rand_flip':[]}
        return np.array(landmrk_msk_lst)
    
    
    *******************************************************************************
    ################# FACE RECOGNITION DATA_LOADER END / > ########################
    *******************************************************************************
    """
    
    '''
    def ld_data_gen(self, batch_size=1, fliplr=False, include_msk=False, shuffled=True, ext=None):
        ## glob file name with reg_exp.
        def glob_reg_exp(exp=r"/*", invert=False):
            re_template = re.compile(exp)
            serch_dir = self.data_src_path
            
            if invert is False:
                file_names = [ x for x in os.listdir( serch_dir) if re_template.search(x)]
            else:
                file_names = [ x for x in os.listdir( serch_dir ) if not re_template.search(x)]
                
            map_gen = map(lambda x: os.path.join(serch_dir, x), file_names)
            file_paths = [ path for path in map_gen]
            return file_paths
            
        ## generator of loading training image.
        def get_batch_img(imgs_path):
            iteration = len(imgs_path)//batch_size
            itr_obj = iter(imgs_path)
            
            [h, w] = self.hr_img_size
            low_h, low_w = int(h / self.scalr), int(w / self.scalr)
            
            for _ in range(iteration):  ## load one img per next()
                filNam, imgs, hr_imgs, lr_imgs = [], [], [], []
                batch_imgs_path, rand_num = [], []  ## for loading msk with random style.
                
                for idx in range(batch_size):
                    path = next(itr_obj)
                    batch_imgs_path.append(path)
                    img = self.__imread(path)
                    file_name = path.split(os.sep)[-1].split('.')[-2]  # get file name 
                    
                    ## Default interp -> bilinear
                    img_hr = scipy.misc.imresize(img, self.hr_img_size)
                    img_lr = scipy.misc.imresize(img_hr, (low_h, low_w))
                
                    # If training => do random flip
                    rand_num.append(np.random.random())
                    if fliplr and rand_num[idx] < 0.5:
                        img_hr, img_lr, img = \
                            np.fliplr(img_hr), np.fliplr(img_lr), np.fliplr(img)
        
                    hr_imgs.append(img_hr)
                    lr_imgs.append(img_lr)
                    imgs.append(img)
                    filNam.append(file_name)
                    
                ## For asynchronous loading :
                if include_msk:
                    self.info_dict_buffer['msk_tag'] = include_msk
                    self.info_dict_buffer['batch_imgs_path'] = batch_imgs_path
                    self.info_dict_buffer['rand_flip'] = rand_num
                    
                hr_imgs = np.array(hr_imgs) / 127.5 - 1.
                lr_imgs = np.array(lr_imgs) / 127.5 - 1.
                imgs = np.array(imgs) / 127.5 - 1.
          
                yield hr_imgs, lr_imgs, imgs, filNam
        
        ## Setting the limitation of glob range.. (for each class glob 10 image)
        all_path = glob_reg_exp((os.sep + "*.{}".format(ext)))
        
        
        shuffled and np.random.shuffle(all_path) ## already randomization the path
        rnd_path = all_path
        
        ld_datagen = get_batch_img(rnd_path)
        
        return ld_datagen
    
    '''
    