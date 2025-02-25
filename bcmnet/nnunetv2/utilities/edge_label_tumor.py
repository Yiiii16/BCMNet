import torch
import numpy as np
from monai.transforms import LabelToContour

def extract_edges(target):
    # 假设 target 的形状为 (N, C, D, H, W)，N 是样本数，C 是通道数 (肝脏和肿瘤)
    assert target.dim() == 5, f"输入的维度应该是 (N, C, D, H, W)，但得到了 {target.shape}"

    # 合并肝脏和肿瘤标签，将所有大于0的值视为前景
    liver_labels = target[:, 0, :, :, :]
    tumor_labels = target[:, 1, :, :, :]
    liver_labels=liver_labels.float()
    tumor_labels=tumor_labels.float()
    # combined_labels = (liver_labels > 0) | (tumor_labels > 0)  # 逻辑或操作合并标签
    # combined_labels = combined_labels.float()  # 转换为 float

    # 创建用于存放边缘图的张量
    # edges_list = []
    edgesliver_list=[]
    edgestumor_list=[]
    label_to_contour = LabelToContour(kernel_type="Laplace")

    # 遍历批次中的每个样本
    # for i in range(combined_labels.shape[0]):
    #     sample_labels = combined_labels[i].unsqueeze(0)  # 添加通道维度，变为 (1, D, H, W)
    #     edges = label_to_contour(sample_labels)  # 边缘图生成，输出形状为 (1, D, H, W)
    #     edges_list.append(edges)
    for i in range(liver_labels.shape[0]):
        liver_label = liver_labels[i].unsqueeze(0)  # 添加通道维度，变为 (1, D, H, W)
        edgesliver = label_to_contour(liver_label)  # 边缘图生成，输出形状为 (1, D, H, W)
        edgesliver_list.append(edgesliver)
    for i in range(tumor_labels.shape[0]):
        tumor_label = tumor_labels[i].unsqueeze(0)  # 添加通道维度，变为 (1, D, H, W)
        edgestumor = label_to_contour(tumor_label)  # 边缘图生成，输出形状为 (1, D, H, W)
        edgestumor_list.append(edgestumor)

    # # 合并所有样本的边缘图，得到形状为 (N, 1, D, H, W)
    # edges = torch.cat(edges_list, dim=0)
    # # 添加一个通道维度，使形状变为 (N, 1, D, H, W)
    # edges = edges.unsqueeze(1)
    # 合并所有样本的边缘图，得到形状为 (N, 1, D, H, W)
    edgesliver = torch.cat(edgesliver_list, dim=0)
    # 添加一个通道维度，使形状变为 (N, 1, D, H, W)
    edgesliver = edgesliver.unsqueeze(1)
    # 合并所有样本的边缘图，得到形状为 (N, 1, D, H, W)
    edgestumor = torch.cat(edgestumor_list, dim=0)
    # 添加一个通道维度，使形状变为 (N, 1, D, H, W)
    edgestumor = edgestumor.unsqueeze(1)

    return edgesliver,edgestumor

# # 示例 target 张量
# target = torch.randn(2, 2, 64, 192, 192)  # 假设有 2 个样本，2 个通道 (肝脏和肿瘤)
#
# # 提取边缘图
# edges_tensor_liver,edges_tensor_tumor = extract_edges(target)
#
# # 输出边缘图尺寸
# print(f"提取的边缘图尺寸: {edges_tensor_liver.shape}")  # 应该是 (2, 1, 64, 192, 192)
# print(f"提取的边缘图尺寸: {edges_tensor_tumor.shape}")  # 应该是 (2, 1, 64, 192, 192)
