%% Iris Comparison :
%  The extracted feature template will be further used in this stage.
%  The source of template will be divided into 2 type : 
%       Prob Set template -> The input of recognition system.
%       Gallery Set template -> The stored template, which extracted in register phase. 
addpath('./comparison_toolbox');

exp_tag = "b_EDSR.mat";
prob_src_root = "SR_iris_info"; % "SR_iris_info", "HR_iris_info", "LR_iris_info".
prob_src_folder = "behmrk/EDSR"; % except SR method, reserve empty str "".
prob_set_path = fullfile(prob_src_root, prob_src_folder);

gallery_mat_dir = fullfile("iris_information", "Gallery_iris_info", 'exp000.mat');
prob_mat_dir = fullfile("iris_information", prob_set_path, exp_tag);

module_prototyp = "DL_based"; % "GAN_based", "CV_based". 
module_name = "EDSR";  % all be lower case.
sr_path = fullfile(module_prototyp, module_name);
%ori_path = "HR"; % "GT" or "LR"
save_hd_path = fullfile("HD_value", sr_path, "hd_b_EDSR.mat"); 

result = hd_comparison(gallery_mat_dir, prob_mat_dir, save_hd_path);
