from typing import Union, Any

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class _InterpolateCustom(autograd.Function):

    @staticmethod
    def forward(ctx, f_input, params):
        shape, perf, grad_conv = params
        ctx.grad_conv = grad_conv
        ctx.perforation = perf
        ctx.orig_shape = tuple(f_input.shape[-2:])
        return interpolate_keep_values(f_input, (shape[-2], shape[-1]))

    @staticmethod
    def backward(ctx, grad_output):
        # TODO test ker z interpolacijo bi bilo (maybe) bolj natančen backward gradient?
        # TEST THIS ker like idk
        with torch.no_grad():  # if ctx.perforation == "both":
            if ctx.perforation == "first":
                return grad_output[:, :, ::2, :], None
            elif ctx.perforation == "second":
                return grad_output[:, :, :, ::2], None
            elif ctx.perforation == "both":
                if ctx.grad_conv:
                    #TODO nek splošen postopek za dobiti ta kernel za vsak nivo perforacije
                    # pokaži razliko v gradientu med non-perforated, tem in samo vsako drugo vrednostjo
                    return F.conv2d(
                        grad_output.view(grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],
                                         grad_output.shape[3]),  # Gaussian approximation
                        torch.tensor(
                            [[[[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]]])
                        .to(grad_output.device), padding=1, stride=(2, 2)).view(
                        grad_output.shape[0], grad_output.shape[1],
                        grad_output.shape[2] // 2 + (grad_output.shape[2] % 2),
                        grad_output.shape[3] // 2 + (
                                grad_output.shape[3] % 2)), None
                return grad_output[:, :, ::2, ::2], None
            else:
                return grad_output, None
        # Razen če bomo hoteli zaradi nekega razloga brez interpolacije pošiljat naprej gradiente


class InterpolatePerforate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, out_shape, perforation, grad_conv):
        return _InterpolateCustom.apply(x, (out_shape, perforation, grad_conv))


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
                 perforation_mode: str = None,
                 use_custom_interp: bool = False,
                 grad_conv: bool = True) -> None:

        super().__init__()
        self.grad_conv = grad_conv
        self.use_custom_interp = use_custom_interp
        if perforation_mode is not None:
            if not perforation_mode in ["first", "second", "both", "none"]:
                print(f"Perforation mode {perforation_mode} Not currently supported.")
                raise NotImplementedError
            else:
                if self.use_custom_interp:
                    self.inter = InterpolatePerforate()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.perforation = perforation_mode if perforation_mode is not None else "none"

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                              padding_mode, device, dtype)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    # noinspection PyTypeChecker
    def forward(self, x, current_perforation="both"):

        # self.perforation = current_perforation
        current_perforation = self.perforation
        out_x = (x.shape[-2] - 2 * (self.conv.kernel_size[0] // 2) + 2 * self.conv.padding[0]) // self.conv.stride[0]
        out_y = (x.shape[-2] - 2 * (self.conv.kernel_size[0] // 2) + 2 * self.conv.padding[0]) // self.conv.stride[0]
        out_x_2 = (x.shape[-2] - 2 * (self.conv.kernel_size[0] // 2) + 2 * self.conv.padding[0]) // (
                    self.conv.stride[0] * 2)
        out_y_2 = (x.shape[-2] - 2 * (self.conv.kernel_size[0] // 2) + 2 * self.conv.padding[0]) // (
                    self.conv.stride[0] * 2)
        if out_x_2 <= 1:
            if out_y_2 <= 1:
                current_perforation = "none"
            else:
                current_perforation = "second"
        elif out_y_2 <= 1:
            current_perforation = "first"
        #if out_x_2 <= 1 or out_y_2 <= 1:
        #    current_perforation = "none"
        x = F.conv2d(x, self.conv.weight, self.conv.bias,
                     (self.conv.stride[0] * (2 if current_perforation in ["first", "both"] else 1),  # TODO optimise
                      self.conv.stride[1] * (2 if current_perforation in ["second", "both"] else 1)), self.conv.padding,
                     self.conv.dilation, self.conv.groups)
        if current_perforation != "none":
            if self.use_custom_interp:
                x = self.inter(x, (out_x, out_y), current_perforation, self.grad_conv)
            else:
                x = F.interpolate(x, (out_x, out_y), mode="bilinear")
        return x


def interpolate_keep_values(input_tensor, desired_shape, duplicate_edges=True):
    """
    :param duplicate_edges: whether to fill in edge with repeat_edge_pixel, or keep interpolation between first and last row/col
    :param input_tensor:
    :param desired_shape: last two dimensions of the output tensor, only accepts `in_dim * 2` and `(in_dim * 2) - 1` due to a  requirement of keeping the same calculated values. In case of `in_dim * 2` shape, the last value is duplicated due to missing information.
    :return: interpolated tensor of the desired shape
    
    Ideally, the `(in_dim * 2)` outputs should have the edge duplicated, but that costs significant extra time.
    """
    #
    orig_x, orig_y = input_tensor.shape[-2:]
    out_x, out_y = desired_shape
    first = (out_x - orig_x) > 1
    second = (out_y - orig_y) > 1
    dblx = not (orig_x * 2) > out_x
    dbly = not (orig_y * 2) > out_y
    if first and orig_x == 1:
        if second and orig_y == 1:
            return input_tensor.expand(out_x, out_y)

    # Yes, i *could* prettify the code by not duplicating it in every if, but the decision process is more clearly visible
    # and the parameter switching would have been ugly

    if first:  # first dim
        off = torch.roll(input_tensor, 1, dims=-2)
        mids = ((input_tensor + off) / 2)
        if second:  # second dim
            if dblx:
                out_tensor = torch.roll(torch.stack((mids, input_tensor), dim=-2)
                                        .view(input_tensor.shape[0], input_tensor.shape[1], out_x,
                                              orig_x), -1, dims=-2)
                off2 = torch.roll(out_tensor, 1, dims=-1)
                mids2 = ((out_tensor + off2) / 2)
                if dbly:
                    out_tensor = torch.roll(
                        torch.stack((mids2, out_tensor), dim=-1).view(
                            input_tensor.shape[0], input_tensor.shape[1], out_x, out_y), -1, dims=-1)
                    if duplicate_edges:
                        out_tensor[:, :, -1, :] = out_tensor[:, :, -2, :]
                        out_tensor[:, :, :, -1] = out_tensor[:, :, :, -2]
                    return out_tensor
                else:
                    out_tensor = torch.roll(
                        torch.stack((mids2, out_tensor), dim=-1).view(
                            input_tensor.shape[0], input_tensor.shape[1], out_x, orig_y * 2), -1, dims=-1)
                    if duplicate_edges:
                        out_tensor[:, :, -1, :] = out_tensor[:, :, -2, :]
                    return out_tensor[:, :, :, :out_y]
            else:
                out_tensor = torch.roll(torch.stack((mids, input_tensor), dim=-2)
                                        .view(input_tensor.shape[0], input_tensor.shape[1], orig_x * 2,
                                              orig_x), -1, dims=-2)
                off2 = torch.roll(out_tensor, 1, dims=-1)
                mids2 = ((out_tensor + off2) / 2)
                if dbly:

                    out_tensor = torch.roll(
                        torch.stack((mids2, out_tensor), dim=-1).view(
                            input_tensor.shape[0], input_tensor.shape[1], orig_x * 2, out_y), -1, dims=-1)
                    if duplicate_edges:
                        out_tensor[:, :, :, -1] = out_tensor[:, :, :, -2]
                    return out_tensor[:, :, :out_x, :]
                else:
                    return torch.roll(
                        torch.stack((mids2, out_tensor), dim=-1).view(
                            input_tensor.shape[0], input_tensor.shape[1], 2 * orig_x, 2 * orig_y), -1, dims=-1)[:, :,
                           :out_x, :out_y]

        else:
            if dblx:
                out_tensor = torch.roll(
                    torch.stack((mids, input_tensor), dim=-2)
                    .view(input_tensor.shape[0], input_tensor.shape[1], out_x, orig_y), -1, dims=-2)
                return out_tensor
            else:
                out_tensor = torch.roll(torch.stack((mids, input_tensor), dim=-2)
                                        .view(input_tensor.shape[0], input_tensor.shape[1], orig_x * 2, orig_x), -1,
                                        dims=-2)
                if duplicate_edges:
                    out_tensor[:, :, -1, :] = out_tensor[:, :, -2, :]
                return out_tensor[:, :, :out_x, :]

    elif second:
        off2 = torch.roll(input_tensor, 1, dims=-1)
        mids2 = ((input_tensor + off2) / 2)
        if dbly:
            out_tensor = torch.roll(
                torch.stack((mids2, input_tensor), dim=-1)
                .view(input_tensor.shape[0], input_tensor.shape[1], orig_x, out_y), -1, dims=-1)
            out_tensor[:, :, :, -1] = out_tensor[:, :, :, -2]
            return out_tensor
        else:
            out_tensor = torch.roll(
                torch.stack((mids2, input_tensor), dim=-1)
                .view(input_tensor.shape[0], input_tensor.shape[1], orig_y, orig_y * 2), -1, dims=-1)
            return out_tensor[:, :, :, :out_y]

    return input_tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sz = 5
    sz21 = 10
    sz22 = 9
    a = torch.zeros((sz, sz))
    a[1::2, ::2] = 1
    a[::2, 1::2] = 1
    cv = nn.Conv2d(1, 1, 3, stride=2)
    test = cv(a[np.newaxis, np.newaxis, :, :])
    print(test.shape)

    a = a * torch.arange(0, sz * sz, 1).view(sz, sz)
    a = torch.stack((a, a))
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(a[0, :, :], vmin=0, vmax=sz * sz - 1)
    axes[0].set_title("Non-interpolated")
    c = interpolate_keep_values(torch.stack((a, -a)), (sz21, sz22))
    print(c.shape)

    axes[1].imshow(torch.squeeze(c)[0, 0], vmin=0, vmax=sz * sz - 1)
    axes[1].set_title("\"Good\" interpolation")
    d = F.interpolate(torch.stack((a, -a)), (sz21, sz22), mode="bilinear", align_corners=False)
    axes[2].imshow(torch.squeeze(d)[0, 0], vmin=0, vmax=sz * sz - 1)
    axes[2].set_title("Normal interpolation")
    plt.tight_layout()
    plt.show()
