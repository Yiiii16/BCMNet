import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
class Sober(nn.Module):
    def __init__(self, channels):
        super(Sober, self).__init__()
        self.channels = channels

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

        sobel_kernel_x = torch.from_numpy(kernel_x).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.from_numpy(kernel_y).unsqueeze(0).unsqueeze(0)
        sobel_kernel_z = torch.from_numpy(kernel_z).unsqueeze(0).unsqueeze(0)

        sobel_kernel_x = sobel_kernel_x.repeat(self.channels, 1, 1, 1, 1)
        sobel_kernel_y = sobel_kernel_y.repeat(self.channels, 1, 1, 1, 1)
        sobel_kernel_z = sobel_kernel_z.repeat(self.channels, 1, 1, 1, 1)

        self.register_buffer('sobel_kernel_x', sobel_kernel_x.float())
        self.register_buffer('sobel_kernel_y', sobel_kernel_y.float())
        self.register_buffer('sobel_kernel_z', sobel_kernel_z.float())

    def forward(self, x):
        assert x.device == self.sobel_kernel_x.device, "Input and Sobel kernels must be on the same device."

        G_x = F.conv3d(x, self.sobel_kernel_x, stride=1, padding=1, groups=x.size(1))
        G_y = F.conv3d(x, self.sobel_kernel_y, stride=1, padding=1, groups=x.size(1))
        G_z = F.conv3d(x, self.sobel_kernel_z, stride=1, padding=1, groups=x.size(1))

        x = torch.sqrt(G_x ** 2 + G_y ** 2 + G_z ** 2 + 1e-6)
        return x
