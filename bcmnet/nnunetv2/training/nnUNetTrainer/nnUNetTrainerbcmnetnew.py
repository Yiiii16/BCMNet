# trainer_script.py
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from torch import autocast, nn
# trainer_script.py

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
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, pred_edges, true_edges):
        # 将 pred_edges 映射到 [0, 1]
        pred_edges = torch.sigmoid(pred_edges)
        true_edges = true_edges.float()

        # 展平张量，保留批次维度
        pred_edges_flat = pred_edges.view(pred_edges.size(0), -1)
        true_edges_flat = true_edges.view(true_edges.size(0), -1)

        # 计算余弦相似度
        cos_sim = F.cosine_similarity(pred_edges_flat, true_edges_flat, dim=1)
        # 将 cos_sim 从 [-1, 1] 映射到 [-1, 0]
        loss = (cos_sim - 1) / 2


        # 取余弦相似度的平均值作为损失
        loss = -cos_sim.mean()

        return loss



# 定义自定义训练器类
class nnUNetTrainerbcmnetnew(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # 初始化边缘损失函数
        self.edge_loss_fn = EdgeLoss()
        self.use_loss_final=True
        # 设置边缘损失的权重，您可以根据需要调整
        # 定义权重超参数
        self.lambda_l1 = 1.0  # 上边分割损失权重
        self.lambda_l2 = 0.1  # 肝脏边缘损失权重
        self.lambda_l3 = 0.8  # 肿瘤边缘损失权重（加大权重）
        self.lambda_l4 = 1.0  # 融合分割损失权重
        # # 计算总损失
        # l = lambda_l1 * l_1 + lambda_l2 * l_2 + lambda_l3 * l_3

        # 在日志中初始化 'train_edge_losses' 键
        #self.logger.my_fantastic_logging['train_edge_losses'] = []  # 确保这是一个空的列表
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
        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
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
            # # 打印所有参数名称
            # for name, param in model.state_dict().items():
            #     print(name)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        print("UMambaBoundarytumorfusion: {}".format(model))

        return model


    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        #data = data.to(self.device, non_blocking=True)
        data = data.to(self.device, non_blocking=True).float()  # 确保输入为 float32
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True).float() for i in target]  # 转换为 float32
        else:
            target = target.to(self.device, non_blocking=True).float()  # 转换为 float32

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output,output_2,final = self.network(data)
            # del data
            l_1 = self.loss(output, target)
            l_4 = self.base_loss(final, target[0])
            output_2_liver=output_2[:, 0, :, :, :].unsqueeze(1)
            output_2_tumor = output_2[:, 1, :, :, :].unsqueeze(1)
            # print(f"edge_out_liver shape: {output_2_liver.shape}")#调试
            # print(f"edge_out_tumor shape: {output_2_tumor.shape}")#调试
            # print(f"target尺寸")
            # # 打印每个张量的尺寸
            # for idx, tensor in enumerate(target):
            #     print(f"target[{idx}] 尺寸: {tensor.shape}")
            # print(f"output尺寸")
            # # 打印每个张量的尺寸
            # for idx, tensor in enumerate(output):
            #     print(f"target[{idx}] 尺寸: {tensor.shape}")
            # print(f"output_2尺寸")


            #print(output_2.shape)
            edges_tensor_liver,_=extract_edges(target[0])
            _, edges_tensor_tumor = extract_edges(target[0])
            # 假设这里取出边缘图的索引为 0
            #edge_tensor = edges[0]['target_2'].to(self.device)  # 获取边缘图，并转移到设备上
            l_2 = self.ce(output_2_liver, edges_tensor_liver)
            l_3 = self.ce(output_2_tumor, edges_tensor_tumor)
            # print("l_1和l_2和l_3的损失")
            # print(l_1)
            # print(l_2)
            # print(l_3)
            # print(l_4)
            l=self.lambda_l1*l_1+self.lambda_l2*l_2+self.lambda_l3*l_3+self.lambda_l4*l_4
            # print(l)
            # print(l_2.requires_grad)  # 应该输出: True
            # print(l_3.requires_grad)  # 应该输出: True
            # 可视化 output_2 和 edges_tensor
            # print(l_2.grad_fn)  # 应该显示梯度函数，如 <AliasBackward0>
            # print(l_3.grad_fn)  # 应该显示梯度函数，如 <AliasBackward0>

            #self.visualize_tensors(output_2_liver,output_2_tumor,edges_tensor_liver, edges_tensor_tumor,target[0],data)


        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            # print("backward的值")
            # print(l_2.grad)  # 应该显示梯度值，例如 tensor([0.1, 0.2, ...])
            # print(l_3.grad)  # 应该显示梯度值，例如 tensor([0.05, 0.15, ...])
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}



    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        #data = data.to(self.device, non_blocking=True)
        data = data.to(self.device, non_blocking=True).float()  # 确保输入为 float32
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True).float() for i in target]  # 转换为 float32
        else:
            target = target.to(self.device, non_blocking=True).float()  # 转换为 float32

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output, output_2, final = self.network(data)
            del data
            l_1 = self.loss(output, target)
            l_4 = self.base_loss(final, target[0])
            output_2_liver=output_2[:, 0, :, :, :].unsqueeze(1)
            output_2_tumor = output_2[:, 1, :, :, :].unsqueeze(1)
            edges_tensor_liver,_ = extract_edges(target[0])
            _,edges_tensor_tumor = extract_edges(target[0])
            # 假设这里取出边缘图的索引为 0
            #edge_tensor = edges[0]['target_2'].to(self.device)  # 获取边缘图，并转移到设备上

            l_2 = self.ce(output_2_liver, edges_tensor_liver)
            l_3 = self.ce(output_2_tumor, edges_tensor_tumor)
            l=self.lambda_l1*l_1+self.lambda_l2*l_2+self.lambda_l3*l_3+self.lambda_l4*l_4


        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            #print("使用has_regions=t")
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
            #print(f"predicted_segmentation_onehot shape: {predicted_segmentation_onehot.shape}")

        else:
            #print("使用has_regions=f")
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            #print(f"output_seg shape: {output_seg.shape}")
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def visualize_tensors(self, output_2_liver, output_2_tumor, edges_tensor_liver,edges_tensor_tumor,target,data):
        # 将张量转换为 CPU 并去除梯度信息
        #output_2_np = output_2.detach().cpu().numpy()
        output_2_liver_np = output_2_liver.detach().cpu().numpy()
        output_2_tumor_np = output_2_tumor.detach().cpu().numpy()
        edges_tensor_liver_np = edges_tensor_liver.detach().cpu().numpy()
        edges_tensor_tumor_np = edges_tensor_tumor.detach().cpu().numpy()
        target_np=target.detach().cpu().numpy()
        data_np=data.detach().cpu().numpy()

        # 选择要显示的切片索引（例如，中间切片）
        slice_idx = output_2_liver_np.shape[2] // 2  # 深度维度的中间切片

        for batch_idx in range(output_2_liver_np.shape[0]):  # 遍历批次中的样本
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

                # 定义保存路径
                save_dir = '/home/23wjy/segmentation/U-Mamba-main/bianyuan_new_tumor/'
                os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在

                # 生成唯一的时间戳
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                # 保存图像
                save_path = os.path.join(save_dir,
                                         f'visualization_step_sample{batch_idx}_slice{slice_idx}_{timestamp}.png')
                plt.savefig(save_path)
                plt.close(fig)  # 关闭图形以释放内存

        # 清空保存目录
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