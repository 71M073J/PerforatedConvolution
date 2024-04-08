import numpy as np
import torch
import torch.nn.functional as F


def interpolate_keep_values_conv(inp, desired_shape, skip_every=(2, 2)):
    sz21, sz22 = desired_shape
    sz1, sz2 = inp.shape[-2], inp.shape[-1]
    if skip_every == (2, 2):
        inp2 = torch.zeros((inp.shape[0], inp.shape[1], sz21, sz22), device=inp.device)
        inp2[:, :, ::2, ::2] = inp
        del inp
        inp2 = F.conv2d(inp2.view(inp2.shape[0] * inp2.shape[1], 1, inp2.shape[2], inp2.shape[3]),
                        torch.tensor([[[[0.25, 0.50, 0.25],
                                        [0.50, 1.00, 0.50],
                                        [0.25, 0.50, 0.25]]]], device=inp2.device), padding=1).view(
            inp2.shape[0], inp2.shape[1], inp2.shape[2], inp2.shape[3])
    elif skip_every == (3, 3):
        inp2 = torch.zeros((inp.shape[0], inp.shape[1], inp.shape[-2] * 3, inp.shape[-1] * 3), device=inp.device)
        inp2[:, :, ::3, ::3] = inp
        del inp
        if (skip_every[0] * sz1) == sz21:
            o1 = sz21
        else:
            o1 = sz21 - (skip_every[0] * sz1)
        if (skip_every[1] * sz2) == sz22:
            o2 = sz22
        else:
            o2 = sz22 - (skip_every[1] * sz2)
        inp2 = F.conv2d(inp2.view(inp2.shape[0] * inp2.shape[1], 1, inp2.shape[2], inp2.shape[3]),
                        torch.tensor([[[[0.1809, 0.2287, 0.3333, 0.2287, 0.1809],
                                        [0.2287, 0.3617, 0.6666, 0.3617, 0.2287],
                                        [0.3333, 0.6666, 1.0000, 0.6666, 0.3333],
                                        [0.2287, 0.3617, 0.6666, 0.3617, 0.2287],
                                        [0.1809, 0.2287, 0.3333, 0.2287, 0.1809]]]], device=inp2.device),
                        padding=2).view(
            inp2.shape[0], inp2.shape[1], inp2.shape[2], inp2.shape[3])[:, :, :o1, :o2]
    # print("asda")
    # if sz22 != skip_every * sz1:
    #    inp2[:, :, -1, :] = inp2[:, :, -2, :]
    # if sz21 != skip_every * sz2:
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


def kern_bicubic(k):
    raise NotImplementedError


def kern1d(k, device):
    return k - torch.arange(-k + 1, k, dtype=torch.float32, device=device).abs()


def get_lin_kernel(stride, normalised=False, device=None):
    k = (kern1d(stride[0], device)[:, None] @ kern1d(stride[1], device)[None, :])[None, None, :, :] / (
                stride[0] * stride[1])
    if normalised:
        return k / k.sum()
    else:
        return k
def kern_polynomial(k, device, n):
    return ((-torch.arange(-k + 1, k, dtype=torch.float32, device=device).abs() ** n) + k**n)/k**(n-1)

def kern_test(k, device, factor=-1.0):
    kern = kern1d(k, device)
    if k == 1:
        return kern
    f_outer = kern[:k//2] * factor
    kern[:k//2] += f_outer
    kern[k//2:k-1] -= f_outer.flip((0,))

    kern[k:-k//2 + 1] -= f_outer
    kern[-k//2+1:] += f_outer.flip((0,))
    return kern
def get_test_kernel(stride, normalised=False, device=None):
    k = (kern_test(stride[0], device)[:, None] @ kern_test(stride[1], device)[None, :])[None, None, :, :] / (
                stride[0] * stride[1])
    #import matplotlib.pyplot as plt
    #plt.imshow(k[0,0])
    #plt.show()
    if normalised:
        return k / k.sum()
    else:
        return k
def get_quad_kernel(stride, normalised=False, device=None):
    k = (kern_polynomial(stride[0], device, 2)[:, None] @ kern_polynomial(stride[1], device, 2)[None, :])[None, None, :, :] / (
                stride[0] * stride[1])
    if normalised:
        return k / k.sum()
    else:
        return k


def get_bicubic_kernel(stride, normalised=False, device=None):
    raise NotImplementedError

def LoG(k, device, sigma=1):
    return - torch.exp(
        -0.5 * torch.square(torch.linspace(-(k - 1) / 2., (k - 1) / 2., k, device=device)) / (sigma * sigma)) * \
    (1 - torch.square(torch.linspace(-(k - 1) / 2., (k - 1) / 2., k, device=device))) * (
                - 1 / (torch.pi * sigma * sigma * sigma * sigma))
def kern_gauss(k, device, sigma=1):
    k2 = k*2-1
    return torch.exp(
        -0.5 * torch.square(torch.linspace(-(k2 - 1) / 2., (k2 - 1) / 2., k2, device=device)) / (sigma * sigma)) * k


def get_gaussian_kernel(stride, normalised=False, device=None):
    k = (kern_gauss(stride[0], device=device)[:, None] @ kern_gauss(stride[1], device=device)[None, :])[None, None, :, :] / (
                stride[0] * stride[1])
    if normalised:
        return k / k.sum()
    else:
        return k
