from typing import Union, Any

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from conv_functions import interpolate_keep_values_conv, interpolate_keep_values, get_lin_kernel


class _InterpolateCustom(autograd.Function):

    @staticmethod
    def forward(ctx, f_input, params):
        shape, ctx.grad_conv, kind, ctx.perf_stride = params
        ctx.orig_shape = tuple(f_input.shape[-2:])
        with torch.no_grad():
            if kind:
                return interpolate_keep_values_deconv(f_input, (shape[-2], shape[-1]), stride=ctx.perf_stride, duplicate=True)
            else:
                return interpolate_keep_values_deconv(f_input, (shape[-2], shape[-1]), stride=ctx.perf_stride, duplicate=False)
                return interpolate_keep_values_conv(f_input, (shape[-2], shape[-1]), perf_stride=ctx.perf_stride)
                return interpolate_keep_values(f_input, (shape[-2], shape[-1]))

    @staticmethod
    def backward(ctx, grad_output):
        # TODO test ker z interpolacijo bi bilo (maybe) bolj natančen backward gradient?
        # TEST THIS ker like idk
        with torch.no_grad():
            if ctx.grad_conv:
                # todo pokaži razliko v gradientu med non-perforated, tem in samo vsako drugo vrednostjo
                return F.conv2d(
                    grad_output.view(grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],
                                     grad_output.shape[3]),  # bilinear interpolation, but inverse
                    get_lin_kernel(ctx.perf_stride, normalised=True, device=grad_output.device),
                    padding=(ctx.perf_stride[0] - 1, ctx.perf_stride[1] - 1),
                    stride=ctx.perf_stride).view(
                    grad_output.shape[0],
                    grad_output.shape[1],
                    - (grad_output.shape[2] // -ctx.perf_stride[0]),
                    - (grad_output.shape[3] // -ctx.perf_stride[1])
                ), None
            else:
                return grad_output[:, :, ::ctx.perf_stride[0], ::ctx.perf_stride[1]], None


class InterpolateFromPerforate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, out_shape, grad_conv, kind, perf_stride):
        return _InterpolateCustom.apply(x, (out_shape, grad_conv, kind, perf_stride))


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


class PerforatedConv2d(nn.Module):
    """
        Class for perforated convolution.

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
                 grad_conv: bool = True,
                 kind=True,
                 perforation_mode=None,
                 perf_stride=None) -> None:


        super().__init__()
        if perf_stride is None:
            if perforation_mode is not None:
                perf_stride = perforation_mode
        self.grad_conv = grad_conv
        self.inter = InterpolateFromPerforate()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.perf_stride = perf_stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                              padding_mode, device, dtype)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.kind = kind
        self.out_x = 0
        self.out_y = 0
        self.recompute = True

    # noinspection PyTypeChecker
    def forward(self, x):
        if self.recompute:
            tmp = 0
            self.out_x = int((x.shape[-2] - ((self.conv.kernel_size[0] - 1) * self.conv.dilation[0]) + 2 * self.conv.padding[
                    0] - 1) // self.conv.stride[0] + 1)
            tmp_stride1 = self.perf_stride[0] + 1
            while tmp <= 1:
                tmp_stride1 -= 1
                if tmp_stride1 == 0:
                    tmp_stride1 = 1
                    break
                tmp = int((x.shape[-2] - ((self.conv.kernel_size[0] - 1) * self.conv.dilation[0]) + 2 *
                               self.conv.padding[0] - 1) // (self.conv.stride[0] * tmp_stride1) + 1)

            tmp = 0
            self.out_y = int((x.shape[-1] - ((self.conv.kernel_size[1] - 1) * self.conv.dilation[1]) + 2 * self.conv.padding[
                1] - 1) // self.conv.stride[1] + 1)
            tmp_stride2 = self.perf_stride[1] + 1
            while tmp <= 1:
                tmp_stride2 -= 1
                if tmp_stride2 == 0:
                    tmp_stride2 = 1
                    break
                tmp = int((x.shape[-1] - ((self.conv.kernel_size[1] - 1) * self.conv.dilation[1]) + 2 *
                           self.conv.padding[1] - 1) // (self.conv.stride[1] * tmp_stride2) + 1)
            self.perf_stride = (tmp_stride1, tmp_stride2)
            self.recompute = False

        x = F.conv2d(x, self.conv.weight, self.conv.bias,
                     (self.conv.stride[0] * self.perf_stride[0],
                      self.conv.stride[1] * self.perf_stride[1]),
                     self.conv.padding,
                     self.conv.dilation, self.conv.groups)
        if self.perf_stride != (1, 1):
            x = self.inter(x, (self.out_x, self.out_y), self.grad_conv, self.kind, self.perf_stride)
        return x


def interpolate_keep_values_deconv(inp, out_shape, stride, duplicate=False):
    interp = F.conv_transpose2d(inp.view(inp.shape[0] * inp.shape[1], 1, inp.shape[2], inp.shape[3]),
                                get_lin_kernel(stride, device=inp.device), stride=stride, padding=(stride[0] - 1, stride[1] - 1),
                                output_padding=((out_shape[0] - 1) % stride[0], (out_shape[1] - 1) % stride[1])).view(
        inp.shape[0],
        inp.shape[1],
        out_shape[0],
        out_shape[1])
    if duplicate:
        if ((out_shape[0] - 1) % stride[0]) > 0:
            interp[:, :, -((out_shape[0] - 1) % stride[0]):, :] = interp[:, :, -1-((out_shape[0] - 1) % stride[0]), :][:,
                                                                                        :, None, :]
        if ((out_shape[1] - 1) % stride[1]) > 0:
            interp[:, :, :, -((out_shape[1] - 1) % stride[1]):] = interp[:, :, :, -1-((out_shape[1] - 1) % stride[1])][:,
                                                                  :, :, None]
    return interp


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sz1 = 5
    sz2 = 4
    sz21 = 10
    sz22 = 7
    a = torch.ones((sz1, sz2))
    # a[1::2, ::2] = 1
    # a[::2, 1::2] = 1
    cv = nn.Conv2d(1, 1, 3, stride=2)
    test = cv(a[np.newaxis, np.newaxis, :, :])
    print(test.shape)

    a = a * torch.arange(0, sz1 * sz2, 1).view(sz1, sz2)
    a = torch.stack((a, a))
    fig, axes = plt.subplots(1, 4)
    axes[0].imshow(a[0, :, :], vmin=0, vmax=sz1 * sz2 - 1)
    axes[0].set_title("Non-interpolated")
    c = interpolate_keep_values(torch.stack((a, -a)), (sz21, sz22))
    test = nn.ConvTranspose2d(2, 2, 3)
    # interpolate_keep_values_deconv(1,1 , (1, 2))

    print(c.shape)
    inp = torch.stack((a, -a))
    ex, ey = 15, 11
    stride = (3, 3)
    # c2 = interpolate_keep_values_conv(inp, (ex, ey), perf_stride=(3, 3))
    c2 = interpolate_keep_values_deconv(inp, (ex, ey), stride=stride, duplicate=True)
    axes[1].imshow(torch.squeeze(c)[0, 0], vmin=0, vmax=sz1 * sz2 - 1)
    axes[1].set_title("\"Good\" interpolation")
    axes[2].imshow(torch.squeeze(c2)[0, 0], vmin=0, vmax=sz1 * sz2 - 1)
    axes[2].set_title("\"Good\" interpolation with conv")
    d = F.interpolate(torch.stack((a, -a)), (sz21, sz22), mode="bilinear", align_corners=False)
    # axes[3].imshow(torch.squeeze(d)[0, 0], vmin=0, vmax=sz1 * sz2 - 1)
    tmp = torch.full((ex, ey), fill_value=1)
    tmp[::stride[0], ::stride[1]] = a[0]
    axes[3].imshow(tmp, vmin=0, vmax=sz1 * sz2 - 1)
    axes[3].set_title("Normal interpolation")
    plt.tight_layout()
    plt.show()
    fig, axes = plt.subplots(1, 3)
    a = torch.rand((1, sz21, sz22))
    a = F.conv2d(a, torch.tensor(
        [[[[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]]]))
    b = interpolate_keep_values_conv(a[None, 0, ::2, ::2], (a.shape[-2], a.shape[-1]))
    axes[0].imshow(a[0])
    axes[0].set_title("original")
    axes[1].imshow(b[0, 0])
    axes[1].set_title("interpolated")
    plt.show()
