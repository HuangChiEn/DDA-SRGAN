function iris_feature_extractor(data_folder, prb_set_folder, dataset_hier, ...
                                ld_bound, ld_msk, ...
                                save_code_path, intrup_flag)
    addpath('utils');    
    dataset_name = fullfile(data_folder, prb_set_folder);
    % Unpack aid information -
    bound_path_cell = {};
    msk_bnd_path_cell = {};
    if ~isempty(ld_bound)
        [bound_path_cell, ~] = get_dataset_path(data_folder, ld_bound); 
        if ~isempty(ld_msk)
            [msk_bnd_path_cell, ~] = get_dataset_path(data_folder, ld_msk);
        end
    elseif ~isempty(ld_msk)
        [msk_bnd_path_cell, ~] = get_dataset_path(data_folder, ld_msk);
    end
    
    % I. Polar Coordinate transform (including Segmentation and Transformation)
    %==========================================================================
    [irisPolar, msk_bnd] = Iris_Preprocessing(dataset_name, dataset_hier, ...
                                        bound_path_cell, msk_bnd_path_cell);
    interval(intrup_flag);
    
    % II. Mask generation
    %==========================================================================
    maskCell = Mask_Generation(irisPolar, msk_bnd);
    interval(intrup_flag);
    
    % III. Feature extraction :
    %==========================================================================
    feaCell = Feature_Extraction(irisPolar, maskCell);
    interval(intrup_flag);
    
    % setting save path & name of mat file.
    fprintf("saving the feature code..\n");
    save(save_code_path, ...
        'irisPolar', 'maskCell', 'feaCell');
    
    fprintf("The whole of Iris Feature Extraction procedure is done..");
    interval(intrup_flag);
end

