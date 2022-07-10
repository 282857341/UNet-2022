# Note
The main differences of these trainer (```nnUnetTrainerV2_unet2022_acdc, nnUnetTrainerV2_unet2022_isic, nnUnetTrainerV2_unet2022_em, nnUnetTrainerV2_unet2022_synapse_224, nnUnetTrainerV2_unet2022_synapse_320```) lies on the function ```do_split()``` and the ```config```.

One more thing to mention is that we did not use cross-validation, so we only used one fold 0 for all experiments. If you need to run the cross-validation, you can modify the ```do_split()``` according to the original code of nnUNet.