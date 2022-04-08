%% Iris Feature Extraction :
%% Prologue -
%@ The path of data source consists with 
%  "../{root_dir}/{imgs_dir}/{dataset_hier}",  (3 Level path setting)
%       {root_dir} : "dataset" (not change).
%       {imgs_dir} : "gen_data" (not change).
%       {dataset_hier} : 
%           1. SR method -> { {module_name}/{exp_tag} }
%           {module_name} : "MA_SRGAN", "n_ESRGAN_p", "SCAST_GAN".
%           {exp_tag} : tag of experiments "exp{num}_CASIA_lab",
%               {num} : "000", "001", "002", ... .
%           2. GT&LR&Gallery -> {folder_name} : "gt_lr_imgs", "reg_iris".
addpath('./extractor_toolbox');

% Top Level - dir name(not change). 
root_dir = "dataset";

% Second Level - folder name of image.
%%  "gen_data" -> SR generated data of Prob set; 
%%  "reg_info" -> Gallery set ;
%%  "gt_lr_img" -> GT&LR data of Prob set;  
imgs_dir = "gen_data";

% Third Level - SR hierachical structure of path.
module_name = "behmrk/EDSR";
exp_tag = "exp001_CASIA_lab";
dataset_hier = fullfile(module_name, exp_tag);
% Ground Truth and Low Resolution.
%dataset_hier = "lr_iris";

ld_bound = fullfile("msk_info", "prb_msk");
ld_msk =  fullfile("msk_info", "prb_msk_img");
%save_code_path = fullfile("iris_information", "SR_iris_info", "SWCA_GAN", ...
%                            "exp001.mat");
save_code_path = fullfile("iris_information", "SR_iris_info", "behmrk", ...
                            "EDSR", "b_EDSR.mat");
build_in_seg = false;

% Execute Iris Feature Extraction :
iris_feature_extractor(root_dir, imgs_dir, dataset_hier, ...
                        ld_bound, ld_msk, save_code_path, false);
