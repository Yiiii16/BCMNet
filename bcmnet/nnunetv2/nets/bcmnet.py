import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from nnunetv2.nets.utils.CBAM import CBAM3D

class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
    ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    #禁用自动混合精度（AMP），强制该层的计算以 float32 精度执行。这在一些情况下对于避免数值不稳定可能是有用的。
    @autocast(enabled=False)
    def forward(self, x):
        #将输入的 x 从 float16 转换为 float32，以确保计算的稳定性。
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        #获取批量大小 (B) 和通道数 (C)。
        B, C = x.shape[:2]
        #确保输入的通道数与预期的维度 self.dim 相符。
        assert C == self.dim
        #计算输入特征图除了 B 和 C 以外的总元素数（相当于展平后的 token 数）。
        n_tokens = x.shape[2:].numel()
        #获取输入特征图的空间维度。
        img_dims = x.shape[2:]
        #将输入特征图展平，并转置维度，使通道维度变为最后一个维度，以适应后续的 Mamba 模块操作。
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        #对展平后的特征图应用 LayerNorm。
        x_norm = self.norm(x_flat)
        #将归一化后的特征图传入 Mamba 模块进行处理。
        x_mamba = self.mamba(x_norm)
        #将处理后的特征图转置回来，并恢复原始的空间维度。
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        #返回经过 Mamba 模块处理后的输出特征图。
        return out



# BasicResBlock 是一个实现基本残差块（Residual Block）的自定义 PyTorch 模块。
# 残差块是 ResNet（Residual Network）架构的核心组件，用于构建深度神经网络，并通过跳跃连接（skip connections）缓解深层网络中的梯度消失问题。
class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
    ):
        super().__init__()

        # 卷积、归一化、激活
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        # norm_op:这是归一化层的构造函数。可以是 PyTorch 中的 nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm 等。它决定了使用哪种归一化方法。
        ##output_channels:这个参数指定了要归一化的特征图的通道数。因为归一化通常是在通道维度上进行的，所以 output_channels 是归一化层所需的关键参数。
        ##norm_op_kwargs:这是一个包含额外参数的字典，传递给归一化层的构造函数。例如，对于 BatchNorm2d，可能会包含 momentum, affine, track_running_stats 等参数。
        ##这行代码的主要作用是实例化一个归一化层，并将其赋值给 self.norm1，以便在 forward 方法中使用。这层归一化的目的是调整卷积输出的激活分布，使得每一层的输出都在一个相对稳定的范围内，这对于加速网络训练和防止梯度爆炸或消失都有帮助。
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        # use_1x1conv:一个布尔值，用于决定是否在跳跃连接上使用 1x1 卷积。如果 input_channels 和 output_channels 不同，或者步幅不是 1 时，通常需要使用 1x1 卷积来匹配尺寸。
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    # ipad图1-1
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class UNetResEncoder(nn.Module):
    def __init__(self,
                 # 输入特征图的通道数。
                 input_channels: int,
                 # 编码器中的阶段数，每个阶段通常对应于不同的分辨率级别。
                 n_stages: int,
                 # 每个阶段的输出通道数，可以是单个整数（表示每个阶段的输出通道数相同），也可以是一个列表（为每个阶段指定不同的输出通道数）。
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 # 卷积操作的类型，通常为 nn.Conv2d 或 nn.Conv3d
                 conv_op: Type[_ConvNd],
                 # 每个阶段的卷积核大小，可以是单个整数或列表。
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 # 每个阶段的步幅，用于控制特征图的下采样程度。
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 # 每个阶段中的残差块数量。
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 # 是否在卷积层中使用偏置项。
                 conv_bias: bool = False,
                 # 归一化操作及其参数，用于卷积层之后的归一化处理。
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 # 非线性激活函数及其参数。
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 # 是否返回每个阶段的输出，通常用于 U-Net 的跳跃连接。
                 return_skips: bool = False,
                 stem_channels: int = None,
                 # 池化类型，如果设置为 'conv'，则使用卷积进行下采样。
                 pool_type: str = 'conv',
                 ):
        super().__init__()

        # 通过这些判断和转换，代码确保了输入参数的一致性和正确性，这使得后续代码中可以统一地处理每个阶段的配置，而不用担心参数类型的不一致。
        # 如果 kernel_sizes 是一个整数，那么将其转换为长度为 n_stages 的列表，每个元素都等于 kernel_sizes。这样，每个阶段都将使用相同大小的卷积核。
        if isinstance(kernel_sizes, int):
            # 如果kernel_sizes = 3-->kernel_sizes = [3, 3, 3, 3]
            # 如果kernel_sizes = [3, 5, 3, 7]->保持不变
            kernel_sizes = [kernel_sizes] * n_stages
        # 这意味着每个阶段的特征图通道数将相同。
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        # 残差块
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        # 步幅
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                  "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"
        # pool_op 是根据卷积操作 conv_op 和 pool_type 确定是否使用池化操作。如果 pool_type 不是 'conv'，则调用 get_matching_pool_op 获取对应的池化操作，否则 pool_op 设为 None。
        ##如果 pool_type 是 'conv'，意味着作者选择使用卷积操作来代替池化（即使用带有步幅的卷积操作来实现降采样效果）。因此，不再需要额外的池化操作，所以将 pool_op 设为 None。
        # 这种设计允许用户灵活选择传统池化（如 maxpool 或 avgpool）或者通过卷积来实现降采样。
        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None
        self.n_blocks_per_stage=n_blocks_per_stage
        # 初始化一个空列表 self.conv_pad_sizes，用于存储每个卷积核对应的填充大小。
        self.conv_pad_sizes = []
        # 遍历 kernel_sizes 列表，其中 krnl 是每一层的卷积核大小。kernel_sizes 是一个列表，每个元素都是一个包含卷积核大小的列表或元组。
        for krnl in kernel_sizes:
            # 对 krnl 中的每个维度 i 进行整除运算 i // 2，计算出填充大小，并将结果作为一个列表添加到 self.conv_pad_sizes 中。
            ##例如，如果 krnl 是 [3, 3]（即 3x3 卷积核），那么填充大小为 [1, 1]，这样在卷积操作后特征图的尺寸可以保持不变。
            ###将卷积核大小除以 2，并取整数部分（即 i // 2），这是在实现**"same" padding**时常用的策略，确保输出的空间尺寸与输入相同。
            ####举例说明：
            # 假设 kernel_sizes 是一个二维列表，表示不同阶段的卷积核大小：
            # kernel_sizes = [[3, 3], [5, 5], [7, 7]]
            # 那么代码执行时：
            # 遍历第一个卷积核 [3, 3]，计算填充大小为 [3 // 2, 3 // 2]，即 [1, 1]。
            # 遍历第二个卷积核 [5, 5]，计算填充大小为 [5 // 2, 5 // 2]，即 [2, 2]。
            # 遍历第三个卷积核 [7, 7]，计算填充大小为 [7 // 2, 7 // 2]，即 [3, 3]。
            # 最终 self.conv_pad_sizes 将变为：
            # self.conv_pad_sizes = [[1, 1], [2, 2], [3, 3]]
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        # 这行代码的作用是将 stem_channels 变量设置为 features_per_stage 列表的第一个元素的值。stem_channels 代表的是 U-Net 编码器中第一层（即“干”层或“起始”层）的输出通道数。
        ##在 U-Net 结构中，通常有一个“起始”或“干”层（stem layer），该层的任务是接收输入数据，并进行初步的特征提取。stem_channels 定义了这一层的输出通道数，它直接影响到后续网络层的计算和特征提取能力。
        ##通过这行代码，确保了网络的起始层输出的特征图通道数与第一阶段的配置一致，这对于构建深度学习模型时的层间兼容性和正确性非常重要。
        stem_channels = features_per_stage[0]

        # 这段代码为 U-Net 编码器的起始层构建了一个复合层。首先通过一个 BasicResBlock 对输入数据进行初步处理，然后通过多个 BasicBlockD 进一步提取特征。
        # stem_channels 确保了特征图通道数的一致性，使得模型在构建过程中能够正确处理和传递特征数据。
        # 这段代码构建了 U-Net 编码器的起始层（stem layer），它是由一个 nn.Sequential 模块组成的。nn.Sequential 用于顺序地将多个子模块堆叠在一起，形成一个复合层。
        self.stem = nn.Sequential(
            # 构建编码器的第一部分，它会对输入进行初步处理和特征提取。
            BasicResBlock(
                # 指定卷积操作类型
                conv_op=conv_op,
                # 输入通道数
                input_channels=input_channels,
                # 输出通道数
                output_channels=stem_channels,
                # 归一化操作及其参数
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                # 卷积核大小
                kernel_size=kernel_sizes[0],
                # 卷积填充大小
                padding=self.conv_pad_sizes[0],
                # 步幅
                stride=1,
                # 非线性激活函数及其参数。
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                # 是否使用 1x1 卷积。
                use_1x1conv=True
            ),
            # *[]:通过列表推导式构建了多个 BasicBlockD 实例，用于在 stem 层中堆叠多个卷积块。
            *[
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1)  # n_blocks_per_stage[0] - 1 表示在第一阶段中，stem 层的额外卷积块数量。
            ]
        )

        # 在 stem 层定义之后，将 input_channels 设置为 stem_channels，表示接下来阶段的输入通道数。
        # 这样确保了 stem 层的输出直接成为 stages 层的输入，保持通道数一致性。
        input_channels = stem_channels

        stages = []
        mamba_layers = []
        stage_fenzhis = []
        CBAM_layers = []
        stage_hous = []
        # 这段代码的作用是在 UNetResEncoder 类中构建多个阶段（stages），并将每个阶段添加到 stages 列表中。
        # 每个阶段包含一个 BasicResBlock 和若干个 BasicBlockD。
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                ),
                *[
                    BasicBlockD(
                        conv_op = conv_op,
                        input_channels = features_per_stage[s],
                        output_channels = features_per_stage[s],
                        kernel_size = kernel_sizes[s],
                        stride = 1,
                        conv_bias = conv_bias,
                        norm_op = norm_op,
                        norm_op_kwargs = norm_op_kwargs,
                        nonlin = nonlin,
                        nonlin_kwargs = nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1)
                ]
            )
            CBAM_block = CBAM3D(features_per_stage[s])
            CBAM_layers.append(CBAM_block)

            mamba_layers.append(
                MambaLayer(
                    dim=features_per_stage[s]
                )
            )

            # 将构建好的 stage 添加到 stages 列表中。
            stages.append(stage)
            stage_hous.append(BasicResBlock(
                conv_op=conv_op,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                input_channels = features_per_stage[s],
                output_channels = features_per_stage[s],
                kernel_size=kernel_sizes[s],
                padding=self.conv_pad_sizes[s],
                stride=1,
                use_1x1conv=True,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs
            ), )
            # input_channels 更新为当前阶段的 features_per_stage[s]，以便下一个阶段正确接收输入数据。
            """
            示例
            假设 n_stages = 3，features_per_stage = [16, 32, 64]，则：

            第一阶段 (s = 0):

            input_channels 初始为 input_channels 的值。
            features_per_stage[0] = 16 设置为该阶段的输出通道数，因此将 input_channels 更新为 16。
            第二阶段 (s = 1):

            input_channels 更新为 features_per_stage[0] = 16 的值，即 16。
            features_per_stage[1] = 32 设置为该阶段的输出通道数，因此将 input_channels 更新为 32。
            第三阶段 (s = 2):

            input_channels 更新为 features_per_stage[1] = 32 的值，即 32。
            features_per_stage[2] = 64 设置为该阶段的输出通道数，因此将 input_channels 更新为 64。
            总结
            更新 input_channels 为 features_per_stage[s] 是为了确保每个阶段的输入通道数与前一阶段的输出通道数一致，保持网络中不同层次之间的通道数一致性。这种做法保证了特征图在网络中的正确传递和处理。
            """
            input_channels = features_per_stage[s]
        # 将之前创建的所有阶段（stages 列表中的 nn.Sequential 对象）组合成一个 nn.Sequential 模块。
        # 这使得 UNetResEncoder 的前向传播过程中，输入数据会依次通过所有定义的阶段进行处理。
        self.stages = nn.Sequential(*stages)
        self.stage_hous = nn.ModuleList(stage_hous)
        # 保存每个阶段的输出通道数。这个属性可能在网络的其他部分（如解码器或后处理层）中用来确定输入通道数。
        self.output_channels = features_per_stage
        self.CBAM_layers = nn.ModuleList(CBAM_layers)
        self.mamba_layers = nn.ModuleList(mamba_layers)
        # strides 是一个表示每个卷积层步幅的列表或元组，可能包含标量（如 2）或列表（如 [2, 2]）。
        # conv_op 代表卷积操作的类型（如 nn.Conv1d, nn.Conv2d, nn.Conv3d），它决定了卷积操作的维度。
        # maybe_convert_scalar_to_list:
        ##如果 i 是一个标量，则根据 conv_op 的维度，将其转换为相应的列表。例如，nn.Conv2d 会将标量 2 转换为 [2, 2]，而 nn.Conv3d 会将标量 2 转换为 [2, 2, 2]。
        ##如果 i 已经是一个列表或元组，则不会进行任何转换，直接返回。
        """
        示例 1：使用标量步幅值
        假设:
        conv_op = nn.Conv2d
        strides = [1, 2, 3]
        执行过程:
        self.strides = [maybe_convert_scalar_to_list(nn.Conv2d, i) for i in [1, 2, 3]]
        结果:
        self.strides = [
            [1, 1],
            [2, 2],
            [3, 3]
        ]
        示例 2：使用混合步幅值
        假设:

        conv_op = nn.Conv3d
        strides = [2, [1, 2, 1], 3]
        结果:
        self.strides = [
            [2, 2, 2],
            [1, 2, 1],
            [3, 3, 3]
        ]
        """
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]

        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs

        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        # 如果 stem 层存在，则先通过 stem 层
        if self.stem is not None:
            x = self.stem(x)
            # print(f"After stem: {x.shape}")

        # 初始化一个空列表，用于存储每个 stage 的输出
        ret = []
        # 依次通过每个 stage，并将输出存储到 ret 列表中
        ##这里的 for 循环逐个遍历 self.stages 中的每个 stage。
        ## s 是每个 stage，表示当前正在处理的网络层模块。
        ## x=s(x):将输入 x 传递给当前 stage 进行处理，得到新的输出 x。x 在每次迭代时被更新，表示每个 stage 的输出作为下一个 stage 的输入。

        for s in range(len(self.stages)):
            x = self.stages[s](x)
            # print("值：",self.n_blocks_per_stage)
            if self.n_blocks_per_stage[s] == 1:
                x_res=x
                # print(f"After stage {s}: {x.shape}")
                # 将当前 stage 的输出 x 添加到列表 ret 中。ret 中最终会包含每个 stage 的输出，
                # 这些输出可以用于跳跃连接（skip connections）、特征融合等用途。
                x_1 = self.mamba_layers[s](x)
                # print(f"x_1 shape{s}:{x_1.shape}")
                x_2 = self.CBAM_layers[s](x)
                x = x_1 + x_2
                x=self.stage_hous[s](x)
                x=x_res+x
            # print(f"After mamba and concatenation {s}: {x.shape}")
            ret.append(x)
        # 如果 return_skips 为 True，返回所有 stage 的输出
        if self.return_skips:
            return ret
        # 否则，只返回最后一个 stage 的输出
        else:
            return ret[-1]

    # 这段代码定义了一个方法 compute_conv_feature_map_size，用于计算经过网络编码器部分（stem 和 stages）后特征图的大小。
    # input_size:这个参数表示输入特征图的尺寸，通常是一个列表或元组，包含每个维度的大小（如高度、宽度和深度）。
    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            # 如果 stem 存在，这一行代码会调用 stem 层的 compute_conv_feature_map_size 方法，计算并返回经过 stem 处理后的特征图大小，并将结果存储在 output 中。
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            # 如果 stem 不存在，output 被初始化为 np.int64(0)。
            output = np.int64(0)

        # 这里的 for 循环遍历所有的 stages。
        for s in range(len(self.stages)):
            # 对于每个 stage，调用 compute_conv_feature_map_size 方法，计算该 stage 处理后的特征图大小并累加到 output 中。
            """
                        示例说明
            假设你的网络有三个 stage，每个 stage 的处理都改变了特征图的尺寸。假设 input_size 初始为 [128, 128]，并且你在每个 stage 层对特征图尺寸的影响如下：

            第一个 stage 的 compute_conv_feature_map_size 方法返回 [64, 64]
            第二个 stage 的 compute_conv_feature_map_size 方法返回 [32, 32]
            第三个 stage 的 compute_conv_feature_map_size 方法返回 [16, 16]
            在遍历 self.stages 时，output 会依次累加每个 stage 层的结果：

            初始 output 为 0。
            遍历第一个 stage 时，output += [64, 64]，output 变为 [64, 64]。
            遍历第二个 stage 时，output += [32, 32]，output 变为 [96, 96]。
            遍历第三个 stage 时，output += [16, 16]，output 变为 [112, 112]。
            注意，这里的 output 累加的是每个 stage 对特征图尺寸的影响，它不是每层的实际输出尺寸，而是一个累计的尺寸。如果你只关心每一层的最终尺寸，可以直接将 output 设置为每一层的输出，而不是累加。
            """
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            # 这行代码根据当前 stage 的 stride 更新 input_size。stride 决定了特征图的缩小倍率，例如 stride=2 会使特征图尺寸减半
            # 通过 zip(input_size, self.strides[s])，input_size 的每个维度都会与对应的 stride 进行除法运算，从而计算出经过该 stage 后特征图的新的尺寸。
            ##input_size 是当前输入特征图的尺寸（通常是一个列表，表示特征图的空间维度，如 [height, width] 或 [depth, height, width]）。
            ##self.strides[s] 是当前 stage 层的步幅（stride）值，也是一个列表，表示每个维度上的步幅。
            ##zip 函数将 input_size 和 self.strides[s] 中的元素按顺序配对成一个元组，生成一个可迭代的对象。
            ## 例如，如果 input_size = [128, 128]，self.strides[s] = [2, 2]，则 zip(input_size, self.strides[s]) 生成 [(128, 2), (128, 2)]。
            ##这是一个列表推导式，它对 zip(input_size, self.strides[s]) 中的每一对 (i, j) 进行整数除法操作，即 i // j，并将结果存储在一个新列表中。
            # 在前面的例子中，i 代表 input_size 中的元素，j 代表对应的步幅。i // j 表示将当前维度上的输入尺寸除以步幅，得到下采样后的尺寸。例如，128 // 2 = 64。
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        # 最终返回经过所有 stages 处理后的总特征图大小。
        return output

class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):

        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                              "resolution stages - 1 (n_stages in encoder - 1), " \
                                                              "here: %d" % n_stages_encoder

        stages = []
        upsample_layers = []

        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='nearest'
            ))

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=2 * input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                ),
                *[
                    BasicBlockD(
                        conv_op=encoder.conv_op,
                        input_channels=input_features_skip,
                        output_channels=input_features_skip,
                        kernel_size=encoder.kernel_sizes[-(s + 1)],
                        stride=1,
                        conv_bias=encoder.conv_bias,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s - 1] - 1)

                ]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            #print(x.shape)
            x = self.stages[s](x)
            # print("x的形状")
            # print(x.shape)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs

        return r,x

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output

class Sober(nn.Module):
    def __init__(self, channels):
        super(Sober, self).__init__()
        self.channels = channels

        # 定义 Sobel 卷积核的 NumPy 数组
        kernel_x = np.array([
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        ], dtype=np.float32)

        kernel_y = np.array([
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        ], dtype=np.float32)

        kernel_z = np.array([
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
        ], dtype=np.float32)

        # 转换为 Torch 张量，并添加必要的维度
        sobel_kernel_x = torch.from_numpy(kernel_x).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 3, 3, 3)
        sobel_kernel_y = torch.from_numpy(kernel_y).unsqueeze(0).unsqueeze(0)
        sobel_kernel_z = torch.from_numpy(kernel_z).unsqueeze(0).unsqueeze(0)

        # 重复卷积核以匹配输入通道数
        sobel_kernel_x = sobel_kernel_x.repeat(self.channels, 1, 1, 1, 1)  # shape: (channels, 1, 3, 3, 3)
        sobel_kernel_y = sobel_kernel_y.repeat(self.channels, 1, 1, 1, 1)
        sobel_kernel_z = sobel_kernel_z.repeat(self.channels, 1, 1, 1, 1)

        # 注册为缓冲区，这样它们会随模型一起移动到正确的设备
        self.register_buffer('sobel_kernel_x', sobel_kernel_x.float())
        self.register_buffer('sobel_kernel_y', sobel_kernel_y.float())
        self.register_buffer('sobel_kernel_z', sobel_kernel_z.float())

    def forward(self, x):
        # 确保输入和卷积核在同一设备上（可选，因为缓冲区会随模型移动）
        assert x.device == self.sobel_kernel_x.device, "Input and Sobel kernels must be on the same device."

        # 执行深度卷积（groups=x.size(1) 表示每个通道单独卷积）
        G_x = F.conv3d(x, self.sobel_kernel_x, stride=1, padding=1, groups=x.size(1))
        G_y = F.conv3d(x, self.sobel_kernel_y, stride=1, padding=1, groups=x.size(1))
        G_z = F.conv3d(x, self.sobel_kernel_z, stride=1, padding=1, groups=x.size(1))
        #print(f'G_x shape: {G_x.shape}, G_y shape: {G_y.shape}, G_z shape: {G_z.shape}')
        # 计算梯度幅值
        x = torch.sqrt(G_x ** 2 + G_y ** 2 + G_z ** 2 + 1e-6)
        return x


class SoberBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, padding_size=1):
        super(SoberBlock, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, padding=padding_size),
                                        nn.InstanceNorm3d(out_size),
                                        nn.LeakyReLU(inplace=True),)
        self.sober = nn.Sequential(nn.Conv3d(out_size,out_size//2,kernel_size=1),
                                        nn.Conv3d(out_size//2,out_size//2,kernel_size=3, padding=1),
                                        nn.InstanceNorm3d(out_size//2),
                                        nn.LeakyReLU(inplace=True),
                                        Sober(out_size//2),
                                        nn.InstanceNorm3d(out_size//2),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Conv3d(out_size//2,out_size,kernel_size=1))
        self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, padding=padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.LeakyReLU(inplace=True),)
        self.sober_out = nn.Conv3d(out_size*2, out_size,kernel_size=1)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        sober = self.sober(outputs)
        outputs = torch.cat([outputs, sober], dim=1)
        outputs = self.sober_out(outputs)
        outputs = self.conv2(outputs)
        return outputs
class UpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpBlock, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample( scale_factor=2,mode='trilinear', align_corners=True), )

    def forward(self, input):
        return self.dsv(input)
class UNetResDecoder_2(nn.Module):
    """
       基于Sober边缘检测的UNet解码器模块。

       参数：
           encoder: 编码器模块，需包含必要的属性如output_channels, strides等。
           num_classes: 分割的类别数。
           n_conv_per_stage: 每个解码阶段的卷积层数，可以是整数、元组或列表。
           deep_supervision: 是否启用深度监督。
           nonlin_first: 是否在第一层使用非线性激活（当前未使用）。
       """

    def __init__(self,
                 encoder,
                 decoder,
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision: bool,
                 nonlin_first: bool = False):
        super(UNetResDecoder_2, self).__init__()

        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.decoder=decoder

        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, \
            "n_conv_per_stage 必须与编码器的分辨率阶段数 - 1 相同，当前编码器阶段数: %d" % n_stages_encoder

        stages = []
        upsample_layers = []
        seg_layers = []
        cat_layer=[]


        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            # 获取当前阶段用于上采样的步幅（stride），用于确定上采样的尺度。
            stride_for_upsampling = encoder.strides[-s]
            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='nearest'
            ))
            cat_layer.append(SoberBlock(2*input_features_skip,input_features_skip))
            stages.append(UpBlock(input_features_skip, 8))
        self.stages = nn.ModuleList(stages)
        # self.upsample_layers = nn.ModuleList(upsample_layers)
        #self.seg_layers = nn.ModuleList(seg_layers)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.cat_layers=nn.ModuleList(cat_layer)
        # self.edge_out = nn.Conv3d(in_channels, out_channels=1, kernel_size=1)
        # 定义 edge_out_conv
        self.edge_out_conv = nn.Conv3d(in_channels=8 * len(stages),
                                       out_channels=2,
                                       kernel_size=1)
    def forward(self, skips):
        """
        前向传播

        参数：
            skips: 编码器的跳跃连接输出，通常是不同分辨率下的特征图列表。

        返回：
            分割输出。如果启用了深度监督，则返回多个输出；否则返回最终输出。
        """
        lres_input = skips[-1]
        seg_outputs = []
        self.edge=[]

        for s in range(len(self.upsample_layers)):
            # 使用 UnetUp3_CT 进行上采样和拼接卷积处理
            x = self.upsample_layers[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            # 使用 SoberConv 进行卷积处理
            x = self.cat_layers[s](x)

            self.edge.append(self.stages[s](x))

            # 更新输入
            lres_input = x

        # print(f"Output shape: {x.shape}")
        out=x

        # 使用 F.interpolate 确保所有边缘输出的形状一致
        edge_outputs = [F.interpolate(e, size=(64, 192, 192), mode='trilinear', align_corners=True) for e in self.edge]

        # # 打印每个边缘输出的尺寸
        # print("edge_outputs尺寸：")
        # for i, output in enumerate(edge_outputs):
        #     print(f"Output {i} shape: {output.shape}")

        # # 拼接边缘输出
        # edge_out = nn.Conv3d(in_channels=sum(e.shape[1] for e in edge_outputs), out_channels=1, kernel_size=1)
        #print("edge_out尺寸前")
        concatenated = torch.cat(edge_outputs, dim=1)
        # print(f"Concatenated shape before edge_out_conv: {concatenated.shape}")  # 调试
        edge_out = self.edge_out_conv(concatenated)
        # print(f"Edge output shape: {edge_out.shape}")  # 调试

        #print("edge_out尺寸后")
        #print(edge_out.shape)
        return edge_out,out

    def compute_conv_feature_map_size(self, input_size):
        """
        计算卷积特征图的尺寸

        参数：
            input_size: 输入尺寸（例如 [C, D, H, W]）

        返回：
            特征图的总大小
        """
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = 0
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += torch.prod(torch.tensor([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]]),
                                 dtype=torch.int64).item()
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += torch.prod(torch.tensor([self.num_classes, *skip_sizes[-(s + 1)]]), dtype=torch.int64).item()
        return output

class bcmnet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1

        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                    f"resolution stages. here: {n_stages}. " \
                                                    f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = UNetResEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels
        )

        self.mamba_layer = MambaLayer(dim=features_per_stage[-1])

        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)
        self.decoder_2 = UNetResDecoder_2(self.encoder,self.decoder, num_classes, n_conv_per_stage_decoder, deep_supervision)
        # 融合层：将两个解码器的输出拼接后，通过1x1卷积融合
        self.fusion = nn.Conv3d(64, 32, kernel_size=1)
        self.final = nn.Conv3d(32, 2, 1)

        # print("features_per_stage的内容")
        # print(features_per_stage)
    def forward(self, x):
        # 确保输入数据类型为 float32
        if x.dtype == torch.float16:
            x = x.float()  # 将输入转换为 float32
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1])
        out1,out1_1=self.decoder(skips)
        out2,out2_1=self.decoder_2(skips)
        # print("out1_1,out2_1尺寸")
        # print(out1_1.shape)
        # print(out2_1.shape)
        # 特征融合：拼接
        fused = torch.cat((out1_1, out2_1), dim=1)  # 假设dim=1是通道维度
        # 通过融合层
        # print("fused的通道")
        # print(fused.shape)
        fused = self.fusion(fused)
        final=self.final(fused)

        return out1,out2,final

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)


def get_bcmnet_3d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
):
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'bcmnet'
    network_class = bcmnet
    kwargs = {
        'bcmnet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))

    return model