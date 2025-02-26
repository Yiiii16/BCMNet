
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from torch import autocast, nn

from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.nets.bcmnet import get_bcmnet_3d_from_plans
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.utilities.edge_label_tumor import extract_edges
import os
import shutil
import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch._dynamo import OptimizedModule

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from sklearn.model_selection import KFold
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

class nnUNetTrainerbcmnet(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.use_loss_final=True
        self.lambda_l1 = 1.0
        self.lambda_l2 = 0.1
        self.lambda_l3 = 0.8
        self.lambda_l4 = 1.0
        self.ce = nn.BCEWithLogitsLoss(reduction="mean")

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        base_loss = DC_and_BCE_loss({},
                                    {'batch_dice': self.configuration_manager.batch_dice,
                                     'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                    use_ignore_label=self.label_manager.ignore_label is not None,
                                    dice_class=MemoryEfficientSoftDiceLoss)
        self.base_loss = base_loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:


        if len(configuration_manager.patch_size) == 3:
            model = get_bcmnet_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        print("UMambaBoundarytumorfusion: {}".format(model))

        return model


    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        #data = data.to(self.device, non_blocking=True)
        data = data.to(self.device, non_blocking=True).float()  
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True).float() for i in target]  
        else:
            target = target.to(self.device, non_blocking=True).float()  

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output,output_2,final = self.network(data)
            # del data
            l_1 = self.loss(output, target)
            l_4 = self.base_loss(final, target[0])
            output_2_liver=output_2[:, 0, :, :, :].unsqueeze(1)
            output_2_tumor = output_2[:, 1, :, :, :].unsqueeze(1)


            edges_tensor_liver,_=extract_edges(target[0])
            _, edges_tensor_tumor = extract_edges(target[0])
            l_2 = self.ce(output_2_liver, edges_tensor_liver)
            l_3 = self.ce(output_2_tumor, edges_tensor_tumor)
            l=self.lambda_l1*l_1+self.lambda_l2*l_2+self.lambda_l3*l_3+self.lambda_l4*l_4



        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}



    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True).float()
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True).float() for i in target]
        else:
            target = target.to(self.device, non_blocking=True).float()
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output, output_2, final = self.network(data)
            del data
            l_1 = self.loss(output, target)
            l_4 = self.base_loss(final, target[0])
            output_2_liver=output_2[:, 0, :, :, :].unsqueeze(1)
            output_2_tumor = output_2[:, 1, :, :, :].unsqueeze(1)
            edges_tensor_liver,_ = extract_edges(target[0])
            _,edges_tensor_tumor = extract_edges(target[0])

            l_2 = self.ce(output_2_liver, edges_tensor_liver)
            l_3 = self.ce(output_2_tumor, edges_tensor_tumor)
            l=self.lambda_l1*l_1+self.lambda_l2*l_2+self.lambda_l3*l_3+self.lambda_l4*l_4


        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()

        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def visualize_tensors(self, output_2_liver, output_2_tumor, edges_tensor_liver,edges_tensor_tumor,target,data):
        output_2_liver_np = output_2_liver.detach().cpu().numpy()
        output_2_tumor_np = output_2_tumor.detach().cpu().numpy()
        edges_tensor_liver_np = edges_tensor_liver.detach().cpu().numpy()
        edges_tensor_tumor_np = edges_tensor_tumor.detach().cpu().numpy()
        target_np=target.detach().cpu().numpy()
        data_np=data.detach().cpu().numpy()

        slice_idx = output_2_liver_np.shape[2] // 2

        for batch_idx in range(output_2_liver_np.shape[0]):
            if((target_np[batch_idx, 1, slice_idx, :, :]).sum!=0):
                fig, axes = plt.subplots(2, 3, figsize=(10, 5))

                axes[0,0].imshow(output_2_liver_np[batch_idx, 0, slice_idx, :, :], cmap='gray')
                axes[0,0].set_title(f'output_2_liver - Sample {batch_idx} - Slice {slice_idx}')

                axes[0,1].imshow(output_2_tumor_np[batch_idx, 0, slice_idx, :, :], cmap='gray')
                axes[0,1].set_title(f'output_2_tumor - Sample {batch_idx} - Slice {slice_idx}')

                axes[0,2].imshow(edges_tensor_liver_np[batch_idx, 0, slice_idx, :, :], cmap='gray')
                axes[0,2].set_title(f'edges_tensor - Sample {batch_idx} - Slice {slice_idx}')

                axes[1,0].imshow(edges_tensor_tumor_np[batch_idx, 0, slice_idx, :, :], cmap='gray')
                axes[1,0].set_title(f'edges_tensor - Sample {batch_idx} - Slice {slice_idx}')

                axes[1,1].imshow(target_np[batch_idx, 1, slice_idx, :, :], cmap='gray')
                axes[1,1].set_title(f'target - Sample {batch_idx} - Slice {slice_idx}')

                axes[1,2].imshow(data_np[batch_idx, 0, slice_idx, :, :], cmap='gray')
                axes[1,2].set_title(f'data - Sample {batch_idx} - Slice {slice_idx}')

                os.makedirs(save_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                save_path = os.path.join(save_dir,
                                         f'visualization_step_sample{batch_idx}_slice{slice_idx}_{timestamp}.png')
                plt.savefig(save_path)
                plt.close(fig)

        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
