% ROC curve plotting and store the result figure
addpath("visual_tool");

iris_load_cell = {"HD_value/GT/hd_exp000", ...
                    "HD_value/GAN_based/masrgan/hd_exp000", "HD_value/GAN_based/n_esrgan_p/hd_exp000", ...
                        "HD_value/GAN_based/swca_gan/hd_exp000", "HD_value/DL_based/RCAN/hd_b_RCAN", ...
                        "HD_value/CV_based/bicubic/hd_exp000", "HD_value/LR/hd_exp000"};

%face_load_cell = {"HD_value/GT/hd_exp000", ...
 %                   "HD_value/GAN_based/masrgan/hd_exp000", "HD_value/GAN_based/n_esrgan_p/hd_exp000", ...
  %                      "HD_value/GAN_based/swca_gan/hd_exp000", "HD_value/DL_based/RCAN/hd_b_RCAN", ...
   %                     "HD_value/CV_based/bicubic/hd_exp000", "HD_value/LR/hd_exp000"};

    
legnd_str_lst = [''];

plot_params = {'--b', '--g', '-r', '-k', '-c', '-y', '-m'};
%plot_params = {'--b', '--r', '--k', '--g'};

plot_roc_curves(iris_load_cell, legnd_str_lst, plot_params, "iris");


