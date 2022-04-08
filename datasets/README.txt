# The MA-SRGAN will be used into the generation of the iris images and the face images. So the different field dataset will be place into this folder..

# 1. CASIA_lab datase :   
	Total images : 20000, Total class : 2000 (Left and Right eyes, per side with 1000 class) ; for each class contains 10 images.
						
	The training data use all of the Left side eyes.
	The evaluation data are divide into 2 part in each class, the previous rank 5 images will be the Gallary set.
	And the rest 5 images will be the Prob set.

	In the evaluation stage, the 5000 x 5000 confusion matrix will be calculated, with Gallary set contains 5 HR images as registration of recognition system.
	In the other hand, the rest 5 images will be downsampling into lower resolution with 4 scalar as the raw input of the system, and the SRGAN will be
	used to upsampling the raw LR into the SR images (same size as HR).

# 2. CelebA dataset :
	Total images : 202599, Total class : 10177 (very ugly dataset..)
	However, the training/eval set will be arrange into the following specification :
		 Total images : 120000, Total class : 6000 
		 4200 class for training for each class contains 20 images.
		 1800 class for evaluation for each class contains 20 images.

	In the evaluation stage, the previous rank 10 images will be the Gallary set of recognition system, and the rest 10 images will be be downsampling into 
	raw LR input of recognition system with 4 scalar.  


# Note : tmp_data is useless, but i'm not sure it's important or not ?! www
         At the result, i just place it in here~~
