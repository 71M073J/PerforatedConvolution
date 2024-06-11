from typing import Union, Any

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from conv_functions import interpolate_keep_values_conv, interpolate_keep_values, get_lin_kernel, \
    interpolate_keep_values_deconv2


class FuckThisShitException(Exception):
    pass


class _InterpolateCustom(autograd.Function):

    @staticmethod
    def forward(ctx, f_input, params):
        shape, ctx.grad_conv, ctx.offset, ctx.perf_stride, ctx.padding, ctx.forward_pad = params
        #shape, ctx.grad_conv, ctx.offset, ctx.perf_stride, ctx.padding, ctx.forward_pad, ctx.filter = params
        ctx.orig_shape = tuple(f_input.shape[-2:])
        with torch.no_grad():
            return interpolate_keep_values_deconv2(f_input, (shape[-2], shape[-1]), stride=ctx.perf_stride,
                                                   duplicate=True, manual_offset=ctx.offset, padding=ctx.padding,
                                                   #filter=ctx.filter
                   )



            #    return interpolate_keep_values_deconv(f_input, (shape[-2], shape[-1]), stride=ctx.perf_stride,
            #                                          duplicate=False)
            #    return interpolate_keep_values_conv(f_input, (shape[-2], shape[-1]), perf_stride=ctx.perf_stride)
            #    return interpolate_keep_values(f_input, (shape[-2], shape[-1]))

    @staticmethod
    def backward(ctx, grad_output):
        # TODO test ker z interpolacijo bi bilo (maybe) bolj natančen backward gradient?
        # TEST THIS ker like idk
        with torch.no_grad():
            if ctx.grad_conv:
                # todo pokaži razliko v gradientu med non-perforated, tem in samo vsako drugo vrednostjo
                # raise NotImplementedError("zdej je Forward pass vsaj pravilno narejen")
                x = ctx.perf_stride[0] + ctx.padding[0] - 1
                y = ctx.perf_stride[1] + ctx.padding[1] - 1
                return F.conv2d(
                    grad_output.view(grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],
                                     grad_output.shape[3]),
                    # bilinear interpolation, but inverse
                    #ctx.filter,
                    get_lin_kernel(ctx.perf_stride, normalised=False, device=grad_output.device),
                    padding=(x, y),  # (ctx.perf_stride[0] - 1 + ctx.offset[0], ctx.perf_stride[1] - 1 + ctx.offset[1]),
                    stride=ctx.perf_stride).view(
                    grad_output.shape[0],
                    grad_output.shape[1],
                    ctx.orig_shape[-2],
                    ctx.orig_shape[-1]
                ), None
            else:
                return grad_output[:, :, ::ctx.perf_stride[0], ::ctx.perf_stride[1]], None


class InterpolateFromPerforate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, out_shape, grad_conv, offset, perf_stride, padding, forward_pad):
        return _InterpolateCustom.apply(x, (out_shape, grad_conv, offset, perf_stride, padding, forward_pad))
    #def forward(self, x, out_shape, grad_conv, offset, perf_stride, padding, forward_pad):#, filter):
    #    return _InterpolateCustom.apply(x, (out_shape, grad_conv, offset, perf_stride, padding, forward_pad))#, filter))



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
                 perforation_mode=None,
                 perf_stride=None) -> None:

        super().__init__()
        if perf_stride is None:
            if perforation_mode is not None:
                perf_stride = perforation_mode
            else:
                perf_stride = (2, 2)
        self.grad_conv = grad_conv
        self.inter = InterpolateFromPerforate()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.perf_stride = perf_stride
        if type(padding) == str:
            if padding == "same":
                padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                              padding_mode, device, dtype)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.out_x = 0
        self.out_y = 0
        self.n1 = 0
        self.n2 = 0
        self.mod1 = 1
        self.mod2 = 1
        self.recompute = True
        self.calculations = 0
        self.in_shape = None
        self.do_offsets = False
        self.hard_limit = (self.conv.kernel_size[0] == 1 and self.conv.kernel_size[1] == 1)
        #self.filter = None

    def set_perf(self, perf):
        self.perf_stride = perf
        self.recompute = True

    # noinspection PyTypeChecker
    def forward(self, x, epoch_offset=0):
        if x.shape[-2:] != self.in_shape:
            self.in_shape = x.shape[-2:]
            self.recompute = True
        if not self.hard_limit:
            if self.recompute:
                tmp = 0
                self.out_x = int(
                    (x.shape[-2] - ((self.conv.kernel_size[0] - 1) * self.conv.dilation[0]) + 2 * self.conv.padding[
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
                self.out_y = int(
                    (x.shape[-1] - ((self.conv.kernel_size[1] - 1) * self.conv.dilation[1]) + 2 * self.conv.padding[
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

                self.mod1 = ((self.out_x - 1) % self.perf_stride[0]) + 1
                self.mod2 = ((self.out_y - 1) % self.perf_stride[1]) + 1
                self.recompute = False
                # in_channels * out_channels * h * w * filter_size // stride1 // stride2
                self.calculations = ((self.conv.in_channels * self.conv.out_channels *
                                      (x.shape[-2] - self.conv.kernel_size[0] // 2 * 2 + self.conv.padding[0] * 2) *
                                      (x.shape[-1] - self.conv.kernel_size[1] // 2 * 2 + self.conv.padding[1] * 2) *
                                      self.conv.kernel_size[0] * self.conv.kernel_size[1]) //
                                     self.conv.stride[0]) // self.conv.stride[1] // \
                                    self.perf_stride[0] // self.perf_stride[1], \
                    f"{self.conv.in_channels}x" \
                    f"{(x.shape[-2] - self.conv.kernel_size[0] // 2 * 2 + self.conv.padding[0] * 2)}x" \
                    f"{(x.shape[-1] - self.conv.kernel_size[1] // 2 * 2 + self.conv.padding[1] * 2)}x" \
                    f"{self.conv.out_channels}x{self.conv.kernel_size[0]}x{self.conv.kernel_size[1]}//" \
                    f"{self.conv.stride[0]}//{self.conv.stride[1]}//{self.perf_stride[0]}//{self.perf_stride[1]}"

                #self.filter = get_lin_kernel(self.perf_stride, normalised=False, device=x.device, dtype=x.dtype)
        else:
            self.perf_stride = (1,1)
        self.recompute = False
        # in_channels * out_channels * h * w * filter_size // stride1 // stride2
        self.calculations = ((self.conv.in_channels * self.conv.out_channels *
                              (x.shape[-2] - self.conv.kernel_size[0] // 2 * 2 + self.conv.padding[0] * 2) *
                              (x.shape[-1] - self.conv.kernel_size[1] // 2 * 2 + self.conv.padding[1] * 2) *
                              self.conv.kernel_size[0] * self.conv.kernel_size[1]) //
                             self.conv.stride[0]) // self.conv.stride[1] // \
                            self.perf_stride[0] // self.perf_stride[1], \
            f"{self.conv.in_channels}x" \
            f"{(x.shape[-2] - self.conv.kernel_size[0] // 2 * 2 + self.conv.padding[0] * 2)}x" \
            f"{(x.shape[-1] - self.conv.kernel_size[1] // 2 * 2 + self.conv.padding[1] * 2)}x" \
            f"{self.conv.out_channels}x{self.conv.kernel_size[0]}x{self.conv.kernel_size[1]}//" \
            f"{self.conv.stride[0]}//{self.conv.stride[1]}//{self.perf_stride[0]}//{self.perf_stride[1]}"
        # raise FuckThisShitException("NEKAJ NE DELA IN NE VEM KAJ")
        if self.perf_stride != (1, 1):
            # raise FuckThisShitException("padding je narobe za stride > 2")
            padding = (self.mod1 - self.n1) % self.mod1, (self.mod2 - self.n2) % self.mod2
            forward_pad = (self.conv.padding[0] + padding[0],
                           self.conv.padding[1] + padding[1])
            x = F.conv2d(x, self.conv.weight, self.conv.bias,
                         (self.conv.stride[0] * self.perf_stride[0], self.conv.stride[1] * self.perf_stride[1]),
                         forward_pad, self.conv.dilation, self.conv.groups)
            # TODO FIX THIS (padding for offset start
            x = self.inter(x, (self.out_x, self.out_y), self.grad_conv, (self.n1, self.n2), self.perf_stride,
                           padding, x.shape[-2:])#, self.filter)

            if self.do_offsets:
                self.n1 = (self.n1 + 1) % self.mod1
                if self.n1 == 0:
                    self.n2 = (self.n2 + 1) % self.mod2  # legit offseti
        else:
            x = F.conv2d(x, self.conv.weight, self.conv.bias,
                         (self.conv.stride[0] * self.perf_stride[0], self.conv.stride[1] * self.perf_stride[1]),
                         self.conv.padding, self.conv.dilation, self.conv.groups)
        return x


def test():
    print("TODOTODOTODO important! naredi verzijo kjer namesto da se gradient downscalea, se direktno preko konvolucije pošlje?")
    print("tesst with differrent train/eval perforation combinations, also test with non-normalised backwards interpo kernel")
    #done, except normalised vs not comparison
    print("TODO test if gradient magnitudes are caused by normalised kernel in backwards for interpolation gradient aggregating")
    print("test if it works with any forward padding")
    conv = PerforatedConv2d(2, 2, 1, padding="same")
    x = torch.ones((1, 2, 8, 8))
    conv.set_perf((2, 2))
    torch.set_printoptions(linewidth=999)
    conv.conv.weight = nn.Parameter(torch.ones_like(conv.conv.weight))
    conv.conv.bias = nn.Parameter(torch.zeros_like(conv.conv.bias))
    x[:, :, 3:, :] = -x[:, :, 3:, :]
    for i in range(5):
        h = conv(x)
        l = h.sum()
        l.backward()
    h = conv(x)
    l = h.sum()
    l.backward()
    conv.zero_grad()
    h = conv(x)
    l = h.sum()
    l.backward()
    print("test")

    quit()


if __name__ == "__main__":
    print("CHECK IF THE BACKWARD INDEXING IS CORRECTLY OFFSET FOR GRADIENTS")
    print("ALSO CHECK IF THE TEST net.eval and net.train fuck with the indexing mybe")
    print("\n\n")
    print("SHIT PERFORMANCE CAUSED BY 1x1 EVAL MODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print("\n\n")
    test()
    import matplotlib.pyplot as plt
    import cv2
    import os
    from conv_functions import get_gaussian_kernel, get_test_kernel

    # assert os.file.exists("./in_channels_2.png")
    im = torch.tensor(cv2.imread("../in_channels_2.png"), dtype=torch.float32).transpose(0, 2).transpose(1,
                                                                                                         2).unsqueeze(0)
    # im = torch.tensor(cv2.cvtColor(cv2.resize(cv2.imread("../landscape.jpg"), (0,0), fx=0.1, fy=0.1), cv2.COLOR_BGR2RGB), dtype=torch.float32).transpose(0,2).transpose(1,2).unsqueeze(0)
    dims = im.shape
    a, b = 1, 1
    step = 4
    fig, axes = plt.subplots(a, b * 2, figsize=(10, 5))
    try:
        axes[0][0]
    except:
        axes = [axes]
    for ind1, i in enumerate(range(1, 1 + a * step, step)):
        for ind2, j in enumerate(range(1, 1 + b * step, step)):
            z = torch.full_like(im, 255.)
            z[:, :, 1::5, 1::5] = im[:, :, 1::5, 1::5]
            axes[ind1][ind2 * 2 + 1].imshow(z.squeeze().transpose(0, 2).transpose(0, 1) / 255.001)
            # im = im[:, :, ::i, ::j+1]
            # im = torch.cat((torch.zeros((im.shape[0], im.shape[1], im.shape[2]//2, 1)),
            #                im.view(im.shape[0], im.shape[1], im.shape[2]//2, -1),
            #                torch.zeros((im.shape[0], im.shape[1], im.shape[2]//2, 1))), dim=-1).view(im.shape[0], im.shape[1], im.shape[-2], -1)
            a = interpolate_keep_values_deconv2(im[:, :, ::5, ::5], (dims[0], dims[1], dims[2], dims[3]), stride=(5, 5),
                                                duplicate=True).squeeze() / 255.00001
            # a = a.view(dims[1], dims[2]//2, -1)[:, :, 1:-1].reshape(dims[1], dims[2], -1)
            print(a.shape, a.transpose(0, 2).shape)
            axes[ind1][ind2 * 2].imshow(a.transpose(0, 2).transpose(0, 1))
    plt.show()
