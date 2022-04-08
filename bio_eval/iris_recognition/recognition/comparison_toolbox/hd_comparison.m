function results = hd_comparison(gallery_mat_dir, prob_mat_dir, save_hd_path)
    %% Prerpocess the gallery set images.
    load(gallery_mat_dir);
    
    len = length(maskCell);
    siz = length(maskCell{1});
    tmp_siz = len*siz;
    gallery_iris_code = cell(len*siz, 1);
    gallery_iris_mask = cell(len*siz, 1);
    gallery_iris_label = zeros(len*siz, 1);  % record each [iris_code, iris_mask] with class label.
    cnt = 1;
    
    %% Accumulate the iris feature code with cross-class.
    for idx=1:len
        msk_lst = maskCell{idx};
        fea_lst = feaCell{idx};
        siz = length(msk_lst);
        
        for jdx=1:siz
            %  Read iris code and cooresponding mask, 
            %       confirm the data type by explicit declartion.
            gallery_iris_mask{cnt, 1} = logical(msk_lst{jdx});
            gallery_iris_code{cnt, 1} = double(fea_lst{jdx});
            gallery_iris_label(cnt, 1) = idx;
            cnt = cnt + 1;
        end
    end  
    disp(['gallery set image done..']);
    
    %% Prerpocess the prob set images.
    load(prob_mat_dir);
    
    len = length(maskCell);
    siz = length(maskCell{1});
    prob_iris_code = cell(len*siz, 1);
    prob_iris_mask = cell(len*siz, 1);
    prob_iris_label = zeros(len*siz, 1);  % record each [iris_code, iris_mask] with class label.
    cnt = 1;
    
    %% Accumulate the iris feature code with cross-class.
    for idx=1:len
        msk_lst = maskCell{idx};
        fea_lst = feaCell{idx};
        siz = length(msk_lst);
        
        for jdx=1:siz
            %  Read iris code and cooresponding mask, 
            %       confirm the data type by explicit declartion.
            prob_iris_mask{cnt, 1} = logical(msk_lst{jdx});
            prob_iris_code{cnt, 1} = double(fea_lst{jdx});
            prob_iris_label(cnt, 1) = idx;
            cnt = cnt + 1;
        end
    end
    disp(['prob set image done..']);
    
    
    %% For the same resolution comparison.. 
    results = runExpwPredTemplateMask2(struct('shift_range', [-20:2:20]), ...
                                        gallery_iris_code, gallery_iris_mask, gallery_iris_label,...
                                        prob_iris_code, prob_iris_mask, prob_iris_label,...
                                        save_hd_path);
end