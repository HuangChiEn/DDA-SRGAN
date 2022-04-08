function [irisPolar, msk_bnd] = ...
                    Iris_Preprocessing(dataset_name, dataset_hier, ...
                                        bound_path_cell, msk_bnd_path_cell)
    PolarType = 1;
    %@ 1.get function handler of transform method..
    switch PolarType
        case 1
            transform = @Cart2PolarFromCircle;
        case 2
            transform = @Cart2LogPolarFromCircle;
        otherwise
            error('No such method : Polar Coordinate transform..\n');
    end
    
    %@ 2.read iris (image & information) and transform into polar coordinate.
    [img_path_cell, categ_info] = get_dataset_path(dataset_name, dataset_hier);
    len = length(img_path_cell);
    categ_info = [1, categ_info];
    irisPolar = cell(categ_info);
    counter = zeros(categ_info);
    
    msk_bnd = cell(categ_info);
    
    cnt = 1; buff_ptr = 1; tmp_buff = {};
    for idx=1:len

        cls_tag = get_cls_tag(img_path_cell{idx});
        %% write the iris polar info mat file from temporary buffer.
        if (cls_tag(1) ~= cnt)
            irisPolar{cnt} = tmp_buff;
            msk_bnd{cnt} = bnd_buff;
            tmp_buff = {};
            bnd_buff = {};
            cnt = cnt + 1;
            counter(cnt) = buff_ptr;
            buff_ptr = 1;
        end
        
        img = imread(img_path_cell{idx});
        img = norm_chnl(img); % Assert the input image is gray scalar.
        
        %@@ 3. extract circular info along the class (cls%idx_%jdx_%zdx) :
        if  isempty(bound_path_cell)
            cirinfo = irisSegNear(img);
            iris_circle = cirinfo(1:3);   
            pupil_circle = cirinfo(4:6);
        else
            % record (x, y, radius) of 2 boundary.
            tmp = csvread(bound_path_cell{idx});
            iris_circle = tmp(1:3);   
            pupil_circle = tmp(4:6); 
        end
        raw = [iris_circle, pupil_circle];
        cirInfo = calib_cirInfo(img, iris_circle, pupil_circle);
        
        %@@ 4. extract the mask boundary info.
        if ~isempty(msk_bnd_path_cell)
            raw_msk = imread(msk_bnd_path_cell{idx});
        end   

        %@@ 5. polar coordinate transformation :
        % compress multi-dim data into class dim {just one dim}.
        tmp_buff{buff_ptr} = transform(img, cirInfo);  
        bnd_buff{buff_ptr} = transform(raw_msk, raw);
        buff_ptr = buff_ptr + 1;

    end
    % The last loop of record
    irisPolar{cnt} = tmp_buff;
    msk_bnd{cnt} = bnd_buff;  %% for mask boundary record.
    counter(cnt) = buff_ptr;
    
    fprintf("Iris Polar Domain Transformation done."); 
end


function cls_tag = get_cls_tag(path_str)
    path_lst = split(path_str, filesep);
    [tmp, ~] = regexp(path_lst{end}, '\d*','match', 'split');
    len = length(tmp);
    cls_tag = zeros(1, len);
    % from string to number.
    for idx=1:len 
        cls_tag(idx) = str2num(tmp{idx});
    end
end


function img = norm_chnl(img)
    chnl_num = length(size(img));
    switch chnl_num
        case 3
            img = rgb2gray(img);
        case 2
            img = img;
        otherwise
            error('Do not support hyper-channels image (greater than 3).');
    end
end


function cirInfo = calib_cirInfo(img, iris_circle, pupil_circle)
    img_siz = size(img);
    div = 1;
    if (img_siz(1) > 480) || (img_siz(2) > 640)
        warning("The image size is grater than the standard specification, the image will be resize into [480 x 640]");
        img = imresize(img, [480, 640]);
    else
        h_div = 480 / img_siz(1);
        w_div = 640 / img_siz(2);

        if h_div ~= w_div
            error('The shrink percentage should be equal.');
        end
        div = h_div;
    end
    cirInfo = [iris_circle/div, pupil_circle/div];
end
