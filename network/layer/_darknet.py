#
#   Darknet related layers
#   Copyright EAVISE
#
#   modified by mileiston
#   modified by nmj 201901

import logging as log
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Con2dBatchLeaky']


class Con2dBatchLeaky(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    They are executed in a sequential manner.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        leaky_slope (number, optional): Controls the angle of the negative slope of the leaky ReLU; Default **0.1**
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slop=0.1):
        super(Con2dBatchLeaky, self).__init__()
        # 属性
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):  #darknet 每一层卷积都有padding
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)
        self.leaky_slope = leaky_slop  # 激活函数

        # layer
        self.layers = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels,
                                              self.kernel_size, self.stride, self.padding,
                                              bias=False),
                                    nn.BatchNorm2d(self.out_channels),  # eps=1e-6, momentum=0.01
                                    nn.LeakyReLU(self.leaky_slope, inplace=True))

    def __repr__(self):
        s = '{name}({in_channels},{out_channels}, kernel_size={kernel_size},' \
            'stride={stride},padding={padding},' \
            'negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


