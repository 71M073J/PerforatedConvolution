from typing import Union, Any

import torch.nn as nn
import torch
import torch.nn.functional as F
from Architectures.PerforatedConv2d import InterpolateFromPerforate


class InterPerfConv2dBNNActivDropout1x1Conv2dInter(nn.Module):
    """
        Class for perforated convolution, with togglable mid-operations and more efficient depthwise convolution

        If getting a small enough input to have to interpolate from a single value, it will simply reject the
         perforation and act as if no perforation is desired
        """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple[int, int]],
                 stride: Union[int, tuple[int, int]] = 1,
                 padding: Union[str, int, tuple[int, int]] = 0,
                 dilation: Union[int, tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device: Any = None,
                 dtype: Any = None,
                 perf_stride=(2, 2),
                 grad_conv: bool = True,
                 kind=True,
                 inter1=None,
                 conv=None,
                 perf_stride2=(2, 2),
                 bnn=None,
                 activ=nn.ReLU(),
                 dropout=None,
                 x1conv=None,
                 inter2=None,
                 ) -> None:

        super().__init__()
        self.inter1 = inter1
        self.inter2 = inter2
        self.conv = conv
        self.bnn = bnn
        self.dropout = dropout
        self.x1conv = x1conv

        self.grad_conv = grad_conv

        self.inter = InterpolateFromPerforate()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if self.conv is None:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                  padding_mode, device, dtype)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.kind = kind
        self.activ = activ
        self.perf_stride = perf_stride
        self.perf_stride2 = perf_stride2

    def forward(self, x):
        if self.inter1 is not None:
            x = self.inter1(x, (x.shape[-2] * 2, x.shape[-1] * 2), self.grad_conv, self.kind,
                            self.perf_stride)
        # TODO načeloma so te dimenzije vnaprej znane? vsaj nekatere
        if self.inter2 is not None:
            out_x = int((x.shape[-2] - ((self.conv.kernel_size[0] - 1) * self.conv.dilation[0]) + 2 * self.conv.padding[
                0] - 1) // self.conv.stride[0] + 1)
            out_y = int((x.shape[-1] - ((self.conv.kernel_size[1] - 1) * self.conv.dilation[1]) + 2 * self.conv.padding[
                1] - 1) // self.conv.stride[1] + 1)
            out_x_2 = int((x.shape[-2] - ((self.conv.kernel_size[0] - 1) * self.conv.dilation[0]) + 2 *
                           self.conv.padding[0] - 1) // (self.conv.stride[0] * self.perf_stride[0]) + 1)
            out_y_2 = int((x.shape[-1] - ((self.conv.kernel_size[1] - 1) * self.conv.dilation[1]) + 2 *
                           self.conv.padding[1] - 1) // (self.conv.stride[1] * self.perf_stride[1]) + 1)
        if out_x_2 <= 1:
            if out_y_2 <= 1:
                current_perforation = "none"
            else:
                current_perforation = "second"
        elif out_y_2 <= 1:
            current_perforation = "first"
        # if out_x_2 <= 1 or out_y_2 <= 1:
        #    current_perforation = "none"
        x = F.conv2d(x, self.conv.weight, self.conv.bias,
                     (self.conv.stride[0] * self.perf_stride2[0],
                      self.conv.stride[1] * self.perf_stride2[1]),
                     self.conv.padding,
                     self.conv.dilation, self.conv.groups)
        if self.bnn is not None:
            x = self.bnn(x)
        if self.activ is not None:
            x = self.activ(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.x1conv is not None:
            x = self.x1conv(x)
        if self.inter2 is not None:
            x = self.inter2(x, self.perf_stride2, self.grad_conv, self.kind, self.perf_stride2)

        # TODO finish this class and maybe test it

        return x
