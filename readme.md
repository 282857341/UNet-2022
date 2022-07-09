# UNet-2022: Exploring Dynamics in Non-isomorphic Architecture
This repository is the official implementation of [UNet-2022: Exploring Dynamics in Non-isomorphic Architecture](). We use the pipeline of [nnUNet](https://github.com/MIC-DKFZ/nnUNet), and the commands including data preprocessing, training and testing all refer to the form of nnUNet.
# Table of contents  
- [Installation](#Installation) 
- [Data-Preparation](#Data-Preparation)
- [Data-Preprocessing](#Data-Preprocessing)
- [Creating_Yaml](#Creating_Yaml)
- [Training_or_Testing_Command](#Training_or_Testing_Command) 
- [How_to_start_your_custom_training](#How_to_start_your_custom_training) 
# Installation
```
git clone https://github.com/282857341/UNet-2022.git
cd UNet-2022
conda env create -f environment.yml
source activate UNet2022
pip install -e .
```
# Data-Preparation
UNet-2022 is a 2D based network, and all data should be expressed in 2D form with ```.nii.gz``` format. You can download the organized dataset from the [link](https://drive.google.com/drive/folders/1b4IVd9pOCFwpwoqfnVpsKZ6b3vfBNL6x?usp=sharing) or download the original data from the link below. If you need to convert other formats (such as ```.jpg```) to the ```.nii.gz```, you can look up the file and modify the [file](https://github.com/282857341/UNet-2022/blob/master/nnunet/dataset_conversion/Task120_ISIC.py) based on your own datasets.

**Dataset I**
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

**Dataset II**
[The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

**Dataset III**
[ISIC2016](https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main), [PH2](https://www.fc.up.pt/addi/ph2%20database.html)

**Dataset IV**
[EM](https://imagej.net/events/isbi-2012-segmentation-challenge#training-data)

The dataset should be finaly organized as follows:
```
./DATASET/
  ├── nnUNet_raw/
      ├── nnUNet_raw_data/
          ├── Task001_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
              ├── evaulate.py

          ├── Task002_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
              ├── evaulate.py              
          ......
      ├── nnUNet_cropped_data/
  ├── nnUNet_trained_models/
  ├── nnUNet_preprocessed/
```
One thing you should be careful of is that folder imagesTr contains both training set and validation set, and correspondingly, the value of 'numTraining' in dataset.json equals the case number in the imagesTr. The division of the training set and validation set will be done in the trainer located at the path ```nnunet/training/network_training```.

The evaulate.py can be found in the [link](https://drive.google.com/drive/folders/1b4IVd9pOCFwpwoqfnVpsKZ6b3vfBNL6x?usp=sharing) of the organized datasets or you can write it by yourself.
# Data-Preprocessing
Before this step, you should set the environment variables to ensure our framework could know the path of nnUNet_raw_data, nnUNet_preprocessed, and  nnUNet_trained_models. The detailed construction can be found in [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md)
```
nnUNet_plan_and_preprocess -t 1
```

```-t 1``` means the command will preprocess the data in the Task001_ACDC 
# Creating_Yaml
```
python create_yaml.py
```
This command will create network configurations for four datasets. You should load it in the trainer located at ```nnunet/training/network_training```.

# Training_or_Testing_Command
```
bash train_or_test.sh -c 0 -n unet2022_acdc -i 1 -s 0.5 -t true -p true 
```
The above command will run the training command and testing command continuously.

You can download the pre-trained weight based on ImageNet or our model weights from this [link](https://drive.google.com/drive/folders/1F9HnLCzWGqoC4BIQ-pDDlnWkmP9Y98Bj?usp=sharing)

You need to adjust the path after the ``` cd``` command in the ```train_or_test.sh``` by yourself.
- ```-c 0``` refers to the index of your Cuda device.
- ```-n unet2022_acdc```denotes the suffix of the trainer located at ```UNet-2022/nnunet/training/network_training/```. For example, nnUNetTrainerV2_unet2022_acdc refers to ```-n unet2022_acdc```
- ```-i 1``` means the index of the task. For example, Task001 refers to ```-i 1```.
- ```-s 0.5``` means the inference step size, reducing the value tends to bring better performance but longer inference time.
- ```-t true/false`` determines whether to run the training command.
- ```-p true/false``` determines whether to run the testing command.

Before you start the testing, please make sure the model_best.model and model_best.model.pkl exists in the specified path, like this:
```
nnUNet_trained_models/nnUNet/2d/Task001_ACDC/nnUNetTrainerV2_unet2022_acdc/fold_0/model_best.model
nnUNet_trained_models/nnUNet/2d/Task001_ACDC/nnUNetTrainerV2_unet2022_acdc/fold_0/model_best.model.pkl
```
# How_to_start_your_custom_training
Every time you start a new training, you should create a new trainer that is located at ```nnunet/training/network_training``` to differentiate it from other trainers and make it inherited from the ```nnUNetTrainer```. After that, you can import your custom network or modify the experimental settings.
