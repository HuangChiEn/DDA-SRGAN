

     
"""
*******************************************************************************
 IRIS RECOGNITION DATA_LOADER : 
*******************************************************************************
"""

    def load_correspond_mask(self):
        
        def converttostr(input_seq, seperator):
           # Join all the strings in list
           final_str = seperator.join(input_seq)
           return final_str
       
        if self.info_dict_buffer['msk_tag'] == False:
            raise Exception("The corresponding data has not be loaded, you should \
                            call load_data function first")
        imgs_msk = []  
        rand_filp = self.info_dict_buffer['rand_flip']
        for idx, img_path in enumerate(self.info_dict_buffer['batch_imgs_path']):
            path_lst = img_path.split('/')
            file_name_lst = path_lst.pop(-1).split('.')
            file_name = file_name_lst[-2]+"_mask."+file_name_lst[-1] 
            msk_dir = converttostr(path_lst, '/')
            img_msk_path = msk_dir + "/msk/" + file_name
            img_msk = imageio.imread(img_msk_path)
            img_msk = scipy.misc.imresize(img_msk, self.hr_img_size)
            if rand_filp and rand_filp[idx] < 0.5:
                img_msk = np.fliplr(img_msk)
            
            imgs_msk.append(img_msk)
            
        imgs_msk = np.array(imgs_msk)
        
        ## resetting the buffer.
        self.info_dict_buffer = {'msk_tag':False, 'batch_imgs_path':[], 'rand_flip':[]}
        return imgs_msk

    '''The Generator based no duplicate image'''
    
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
    