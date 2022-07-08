#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.UNet2022 import unet2022
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from tqdm import tqdm
import yaml

def loadData(data="config.yaml"):

        with open(data, "r") as f:
            content = f.read()
        yamlData = yaml.load(content,Loader=yaml.FullLoader)
        #print("yamlData_type: ", type(yamlData))
        #print("Hyper_parameters: ", yamlData['Hyper_parameters'])
        return yamlData

class nnUNetTrainerV2_unet2022_synapse_320(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
                         
        yamlData = loadData(data="/home/xychen/jsguo/yaml/Synapse_320.yaml")

        
        self.max_num_epochs = yamlData['Hyper_parameters']['Epochs_num']
        self.initial_lr = yamlData['Hyper_parameters']['Base_learning_rate']
        self.batch_size = yamlData['Hyper_parameters']['Batch_size']
        self.patch_size = yamlData['Hyper_parameters']['Crop_size']
        self.Deep_supervision = yamlData['Hyper_parameters']['Deep_supervision']
        #self.learning_rate_schedule = yamlData['Hyper_parameters']['Learning_rate_schedule']
        self.model_size = yamlData['Hyper_parameters']['Model_size']
        self.num_blocks = yamlData['Hyper_parameters']['Blocks_num']
        self.val_eval_criterion_alpha = yamlData['Hyper_parameters']['Val_eval_criterion_alpha'] # used for validation. dice = alpha * old_dice + (1-alpha) * new_dice
        self.convolution_stem_down = yamlData['Hyper_parameters']['Convolution_stem_down']
        self.window_size = yamlData['Hyper_parameters']['Window_size']
        
        self.pretrain = yamlData['Pretrain']
        self.train_list = yamlData['Train_list']
        self.val_list = yamlData['Val_list']
        self.print_to_log_file('Hyper_parameters:',yamlData['Hyper_parameters'],also_print_to_console=True)
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()
            if self.Deep_supervision:
                ################# Here we wrap the loss for deep supervision ############
                # we need to know the number of outputs of the network
                net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(3)])

                weights = weights / weights.sum()
                self.ds_loss_weights = weights
                # now wrap the loss
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
                ################# END ###################
            
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=[[1,1],[0.5,0.5],[0.25,0.25]] if self.Deep_supervision else None,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,         
                    seeds_train=range(int(self.data_aug_params['num_threads'])),
                    seeds_val=range(int(self.data_aug_params['num_threads'])*2)[-self.data_aug_params['num_threads']//2:]
                )
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        assert self.model_size in ['Tiny','Base','Large'], "error key words, or you can dismiss it and set it by yourself"
        
        if self.model_size == 'Tiny':
            self.embedding_dim = 96
            self.num_heads = [3,6,12,24]
            self.pre_trained_weight = torch.load("/home/xychen/jsguo/weight/convnext_t_3393.model",map_location='cpu')
            
        if self.model_size=='Base':
            self.embedding_dim = 128
            self.num_heads = [4,8,16,32]
            self.pre_trained_weight = torch.load("/home/xychen/jsguo/weight/convnext_base.model",map_location='cpu')

        if self.model_size=='Large':
            self.embedding_dim = 192
            self.num_heads = [6,12,24,48]
            self.pre_trained_weight = torch.load("/home/xychen/jsguo/weight/convnext_large.model",map_location='cpu')
       
        # Don't touch conv_op
        self.network = unet2022(self.patch_size, self.num_input_channels, self.embedding_dim, self.window_size, self.num_heads,
                                        self.convolution_stem_down, self.num_classes, self.num_blocks, self.Deep_supervision, 
                                        conv_op=nn.Conv2d)
       
      
        #from ptflops import get_model_complexity_info
 
        #flops, params = get_model_complexity_info(self.network, (self.num_input_channels,self.patch_size[0],self.patch_size[1]),as_strings=True,print_per_layer_stat=False)
        #self.print_to_log_file("|flops: %s |params: %s" % (flops,params))
       
        if self.pretrain:
            checkpoint=self.pre_trained_weight
            ck={}
            for i in self.network.state_dict():
                if i in checkpoint:
                    # if the key in the pre-trained model is the same with ours model, we load it. If not, we initialize it randomly.
                    # so it's necessary to check the key of the pre-trained model and ours
                    ck.update({i:checkpoint[i]})
                else:
                    ck.update({i:self.network.state_dict()[i]})
            self.network.load_state_dict(ck)
            print('I am using the pre_trained weight!!') 
        
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.initial_lr)
        # we use the fixed learning_rate
        self.scheduler = None
        #self.scheduler = CosineAnnealingLR(self.optimizer, T_max=20)
            
    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        if self.Deep_supervision:
            target = target[0]
            output = output[0]
            
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        exit()
        data_dict=data_generator
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
    
        l.detach()
        loss=l.item()
        del l
        return loss

    def do_split(self):
        #print(self.dataset.keys())
        #print(self.train_list)
        #print(self.val_list)
        tr_keys = [i for i in self.dataset.keys() if i.split('_')[0] in train_list]
        val_keys = [i for i in self.dataset.keys() if i.split('_')[0] in val_list]
        self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr =[]
        self.dataset_val =[]
        with tqdm(tr_keys) as tbar:
            for b in tbar:
            
                if isfile(self.dataset[b]['data_file'][:-4] + ".npy"):
                    a = np.load(self.dataset[b]['data_file'][:-4] + ".npy", 'r')
                else:
                    a = np.load(self.dataset[b]['data_file'])['data']
                self.dataset_tr.append(a)
        with tqdm(val_keys) as tbar:
            for b in tbar:
                if isfile(self.dataset[b]['data_file'][:-4] + ".npy"):
                    a = np.load(self.dataset[b]['data_file'][:-4] + ".npy", 'r')
                else:
                    a = np.load(self.dataset[b]['data_file'])['data']
                self.dataset_val.append(a)

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            #if max(self.patch_size) / min(self.patch_size) > 1.5:
            default_2D_augmentation_params['rotation_x'] = [0, 15. / 360 * 2. * np.pi]
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size
        #self.data_aug_params["do_scale"] =False
        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):

        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
       
        #self.scheduler.step()
        #self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
      
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):

        ds = self.network.do_ds
        self.network.do_ds = self.Deep_supervision
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
