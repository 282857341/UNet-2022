#!/bin/bash

while getopts 'c:n:i:s:t:p' OPT; do
    case $OPT in
        c) cuda=$OPTARG;;
        n) name=$OPTARG;;
		i) task=$OPTARG;;
		s) step=$OPTARG;;
        t) train=$OPTARG;;
        p) predict=$OPTARG;;
		
    esac
done
echo $name	

if ${train}
then
	# If you have more than 4000 images to read, increasing the value of number after 'ulimit -n'
	ulimit -n 4000
	CUDA_VISIBLE_DEVICES=${cuda} nnUNet_train 2d nnUNetTrainerV2_${name} ${task} 0
fi

if ${predict}
then
	cd /home/xychen/new_transformer/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task029_acdc_slice/
	CUDA_VISIBLE_DEVICES=${cuda} nnUNet_predict -i imagesTs -o inferTs/${name}_${step}step -m 2d -t ${task} -chk model_best -tr nnUNetTrainerV2_${name} --num_threads_preprocessing 16 --num_threads_nifti_save 16 --step_size ${step}
	#You can find evaulate.py in the organized dataset downloaded from the google drive or you can write it by yourself.
	python evaulate.py ${name}_${step}step
fi






