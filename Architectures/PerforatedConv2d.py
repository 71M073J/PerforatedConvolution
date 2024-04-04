from typing import Union, Any

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class _InterpolateCustom(autograd.Function):

    @staticmethod
    def forward(ctx, f_input, params):
        shape, ctx.perforation, ctx.grad_conv, kind, ctx.skip_every = params
        ctx.orig_shape = tuple(f_input.shape[-2:])
        with torch.no_grad():
            if kind:
                return interpolate_keep_values_conv(f_input, (shape[-2], shape[-1]), skip_every=ctx.skip_every)
            else:
                return interpolate_keep_values(f_input, (shape[-2], shape[-1]))

    @staticmethod
    def backward(ctx, grad_output):
        # TODO test ker z interpolacijo bi bilo (maybe) bolj natančen backward gradient?
        # TEST THIS ker like idk
        with torch.no_grad():  # if ctx.perforation == "both":
            if ctx.perforation == "first":
                return grad_output[:, :, ::ctx.skip_every[0], :], None
            elif ctx.perforation == "second":
                return grad_output[:, :, :, ::ctx.skip_every[1]], None
            elif ctx.perforation == "both":
                if ctx.grad_conv:
                    # TODO nek splošen postopek za dobiti ta kernel za vsak nivo perforacije
                    # pokaži razliko v gradientu med non-perforated, tem in samo vsako drugo vrednostjo
                    #if ctx.skip_every[0] == ctx.skip_every[1] == 2:
                    return F.conv2d(
                        grad_output.view(grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],
                                         grad_output.shape[3]),  # Gaussian approximation
                        torch.tensor(
                            [[[[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]]], device=grad_output.device)
                        , padding=1, stride=ctx.skip_every).view(
                        grad_output.shape[0], grad_output.shape[1],
                        grad_output.shape[2] // 2 + (grad_output.shape[2] % 2),
                        grad_output.shape[3] // 2 + (
                                grad_output.shape[3] % 2)), None


                return grad_output[:, :, ::ctx.skip_every[0], ::ctx.skip_every[1]], None
            elif ctx.perforation == "trip":
                if ctx.grad_conv:
                    #if ctx.skip_every[0] == ctx.skip_every[1] == 3:
                    return F.conv2d(grad_output.view(grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2], grad_output.shape[3]),
                                    torch.tensor([[[[0.1809, 0.2287, 0.3333, 0.2287, 0.1809],
                                                    [0.2287, 0.3617, 0.6666, 0.3617, 0.2287],
                                                    [0.3333, 0.6666, 1.0000, 0.6666, 0.3333],
                                                    [0.2287, 0.3617, 0.6666, 0.3617, 0.2287],
                                                    [0.1809, 0.2287, 0.3333, 0.2287, 0.1809]]]], device=grad_output.device),
                                    padding=2, stride=ctx.skip_every).view(
                        grad_output.shape[0], grad_output.shape[1],
                        -(grad_output.shape[2] // -3),
                        -(grad_output.shape[3] // -3)), None

                return grad_output[:, :, ::ctx.skip_every[0], ::ctx.skip_every[1]], None
            else:
                return grad_output, None
        # Razen če bomo hoteli zaradi nekega razloga brez interpolacije pošiljat naprej gradiente


class InterpolateFromPerforate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, out_shape, perforation, grad_conv, kind, skip_every):
        if perforation == "trip":
            skip_every = (3,3)
        return _InterpolateCustom.apply(x, (out_shape, perforation, grad_conv, kind, skip_every))


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
                 perforation_mode: str = None,
                 skip_every=(2,2),
                 grad_conv: bool = True,
                 kind=True,
                 inter1=None,
                 conv=None,
                 skip_every2=(2,2),
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
        if perforation_mode is not None:
            if perforation_mode not in ["first", "second", "both", "none"]:
                print(f"Perforation mode {perforation_mode} Not currently supported.")
                raise NotImplementedError
            else:
                self.inter = InterpolateFromPerforate()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.perforation = perforation_mode if perforation_mode is not None else "none"

        if self.conv is None:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                  padding_mode, device, dtype)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.kind = kind
        self.activ = activ
        self.skip_every = skip_every
        self.skip_every2 = skip_every2
    def forward(self, x, current_perforation=None):

        if current_perforation is None:
            current_perforation = self.perforation
        if self.inter1 is not None:
            x = self.inter1(x, (x.shape[-2]*2, x.shape[-1]*2), self.perforation, self.grad_conv, self.kind, self.skip_every)
        # self.perforation = current_perforation
        #TODO načeloma so te dimenzije vnaprej znane? vsaj nekatere
        if self.inter2 is not None:
            out_x = int((x.shape[-2] - ((self.conv.kernel_size[0] - 1) * self.conv.dilation[0]) + 2 * self.conv.padding[
                0] - 1) // self.conv.stride[0] + 1)
            out_y = int((x.shape[-1] - ((self.conv.kernel_size[1] - 1) * self.conv.dilation[1]) + 2 * self.conv.padding[
                1] - 1) // self.conv.stride[1] + 1)
            out_x_2 = int((x.shape[-2] - ((self.conv.kernel_size[0] - 1) * self.conv.dilation[0]) + 2 *
                           self.conv.padding[0] - 1) // (self.conv.stride[0] * self.skip_every[0]) + 1)
            out_y_2 = int((x.shape[-1] - ((self.conv.kernel_size[1] - 1) * self.conv.dilation[1]) + 2 *
                           self.conv.padding[1] - 1) // (self.conv.stride[1] * self.skip_every[1]) + 1)
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
                     (self.conv.stride[0] * (self.skip_every2[0] if current_perforation in ["first", "both", "trip"] else 1),  # TODO optimise
                      self.conv.stride[1] * (self.skip_every2[1] if current_perforation in ["second", "both", "trip"] else 1)), self.conv.padding,
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
            x = self.inter2(x, self.skip_every2, self.perforation, self.grad_conv, self.kind, self.skip_every2)


        #TODO finish this class and maybe test it

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
                 perforation_mode: str = None,
                 grad_conv: bool = True,
                 kind=True,
                 skip_every=(2, 2)) -> None:

        super().__init__()
        self.grad_conv = grad_conv
        if perforation_mode is not None:
            if perforation_mode not in ["first", "second", "both", "none", "trip"]:
                print(f"Perforation mode {perforation_mode} Not currently supported.")
                raise NotImplementedError
            else:
                self.inter = InterpolateFromPerforate()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.perforation = perforation_mode if perforation_mode is not None else "none"
        self.skip_every=skip_every
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                              padding_mode, device, dtype)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.kind = kind

    # noinspection PyTypeChecker
    def forward(self, x, current_perforation=None):

        # self.perforation = current_perforation
        if current_perforation is None:
            current_perforation = self.perforation
        if current_perforation == "both":
            out_x = int((x.shape[-2] - ((self.conv.kernel_size[0]-1) * self.conv.dilation[0]) + 2 * self.conv.padding[0] -1) // self.conv.stride[0] + 1)
            out_y = int((x.shape[-1] - ((self.conv.kernel_size[1]-1) * self.conv.dilation[1]) + 2 * self.conv.padding[1] -1) // self.conv.stride[1] + 1)
            out_x_2 = int((x.shape[-2] - ((self.conv.kernel_size[0]-1) * self.conv.dilation[0]) + 2 * self.conv.padding[0] -1) // (self.conv.stride[0] * self.skip_every[0]) + 1)
            out_y_2 = int((x.shape[-1] - ((self.conv.kernel_size[1]-1) * self.conv.dilation[1]) + 2 * self.conv.padding[1] -1) // (self.conv.stride[1] * self.skip_every[1]) + 1)
            if out_x_2 <= 1:
                if out_y_2 <= 1:
                    current_perforation = "none"
                else:
                    current_perforation = "second"
            elif out_y_2 <= 1:
                current_perforation = "first"
        elif current_perforation == "trip":
            self.skip_every = (3,3)
            out_x = int((x.shape[-2] - ((self.conv.kernel_size[0]-1) * self.conv.dilation[0]) + 2 * self.conv.padding[0] -1) // self.conv.stride[0] + 1)
            out_y = int((x.shape[-1] - ((self.conv.kernel_size[1]-1) * self.conv.dilation[1]) + 2 * self.conv.padding[1] -1) // self.conv.stride[1] + 1)
            out_x_2 = int((x.shape[-2] - ((self.conv.kernel_size[0]-1) * self.conv.dilation[0]) + 2 * self.conv.padding[0] -1) // (self.conv.stride[0] * self.skip_every[0]) + 1)
            out_y_2 = int((x.shape[-1] - ((self.conv.kernel_size[1]-1) * self.conv.dilation[1]) + 2 * self.conv.padding[1] -1) // (self.conv.stride[1] * self.skip_every[1]) + 1)
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
                     (self.conv.stride[0] * (self.skip_every[0] if current_perforation in ["first", "both", "trip"] else 1),  # TODO optimise
                      self.conv.stride[1] * (self.skip_every[1] if current_perforation in ["second", "both", "trip"] else 1)), self.conv.padding,
                     self.conv.dilation, self.conv.groups)
        if current_perforation != "none":
            x = self.inter(x, (out_x, out_y), current_perforation, self.grad_conv, self.kind, self.skip_every)
        return x


def interpolate_keep_values_conv(inp, desired_shape, skip_every=(2,2)):
    sz21, sz22 = desired_shape
    sz1, sz2 = inp.shape[-2], inp.shape[-1]
    if skip_every == (2,2):
        inp2 = torch.zeros((inp.shape[0], inp.shape[1], sz21, sz22), device=inp.device)
        inp2[:, :, ::2, ::2] = inp
        del inp
        inp2 = F.conv2d(inp2.view(inp2.shape[0] * inp2.shape[1], 1, inp2.shape[2], inp2.shape[3]),
                      torch.tensor([[[[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]]], device=inp2.device), padding=1).view(
            inp2.shape[0], inp2.shape[1], inp2.shape[2], inp2.shape[3])
    elif skip_every == (3,3):
        inp2 = torch.zeros((inp.shape[0], inp.shape[1], inp.shape[-2]*3, inp.shape[-1]*3), device=inp.device)
        inp2[:, :, ::3, ::3] = inp
        del inp
        if (skip_every[0]*sz1) == sz21:
            o1 = sz21
        else:
            o1 = sz21 - (skip_every[0]*sz1)
        if (skip_every[1]*sz2) == sz22:
            o2 = sz22
        else:
            o2 = sz22 - (skip_every[1]*sz2)
        inp2 = F.conv2d(inp2.view(inp2.shape[0] * inp2.shape[1], 1, inp2.shape[2], inp2.shape[3]),
                        torch.tensor([[[[0.1809, 0.2287, 0.3333, 0.2287, 0.1809],
                                          [0.2287, 0.3617, 0.6666, 0.3617, 0.2287],
                                          [0.3333, 0.6666, 1.0000, 0.6666, 0.3333],
                                          [0.2287, 0.3617, 0.6666, 0.3617, 0.2287],
                                          [0.1809, 0.2287, 0.3333, 0.2287, 0.1809]]]], device=inp2.device),
                        padding=2).view(
            inp2.shape[0], inp2.shape[1], inp2.shape[2], inp2.shape[3])[:, :, :o1, :o2]
    #print("asda")
    #if sz22 != skip_every * sz1:
    #    inp2[:, :, -1, :] = inp2[:, :, -2, :]
    #if sz21 != skip_every * sz2:
    #    inp2[:, :, :, -1] = inp2[:, :, :, -2]
    return inp2


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
                                              orig_y), -1, dims=-2)
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
                                        .view(input_tensor.shape[0], input_tensor.shape[1], orig_x * orig_x), -1,
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

    sz1 = 5
    sz2 = 4
    sz21 = 10
    sz22 = 7
    a = torch.zeros((sz1, sz2))
    a[1::2, ::2] = 1
    a[::2, 1::2] = 1
    cv = nn.Conv2d(1, 1, 3, stride=2)
    test = cv(a[np.newaxis, np.newaxis, :, :])
    print(test.shape)

    a = a * torch.arange(0, sz1 * sz2, 1).view(sz1, sz2)
    a = torch.stack((a, a))
    fig, axes = plt.subplots(1, 4)
    axes[0].imshow(a[0, :, :], vmin=0, vmax=sz1 * sz2 - 1)
    axes[0].set_title("Non-interpolated")
    c = interpolate_keep_values(torch.stack((a, -a)), (sz21, sz22))
    print(c.shape)
    inp = torch.stack((a, -a))
    ex, ey = 15, 12
    c2 = interpolate_keep_values_conv(inp, (ex, ey), skip_every=(3,3))
    axes[1].imshow(torch.squeeze(c)[0, 0], vmin=0, vmax=sz1 * sz2 - 1)
    axes[1].set_title("\"Good\" interpolation")
    axes[2].imshow(torch.squeeze(c2)[0, 0], vmin=0, vmax=sz1 * sz2 - 1)
    axes[2].set_title("\"Good\" interpolation with conv")
    d = F.interpolate(torch.stack((a, -a)), (sz21, sz22), mode="bilinear", align_corners=False)
    #axes[3].imshow(torch.squeeze(d)[0, 0], vmin=0, vmax=sz1 * sz2 - 1)
    tmp = torch.full((ex, ey), fill_value=1)
    tmp[::3, ::3] = a[0]
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
    axes[1].imshow(b[0,0])
    axes[1].set_title("interpolated")
    plt.show()