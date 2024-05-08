from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Sequential
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from .PerforatedConv2d import PerforatedConv2d
from .conv2dNormActivation import Conv2dNormActivation
import numpy as np

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1,
            perforation_mode: tuple = (1, 1), grad_conv: bool = True) -> PerforatedConv2d:
    """3x3 convolution with padding"""
    return PerforatedConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        perforation_mode=perforation_mode,
        grad_conv=grad_conv
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, perforation_mode: tuple = (1, 1),
            grad_conv: bool = True
            ) -> PerforatedConv2d:
    """1x1 convolution"""
    return PerforatedConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                            perforation_mode=perforation_mode, grad_conv=grad_conv)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None, perforation_mode: list = None,
            grad_conv: bool = True
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, perforation_mode=perforation_mode[0], grad_conv=grad_conv)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, perforation_mode=perforation_mode[1], grad_conv=grad_conv)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            perforation_mode: list = None,
            grad_conv: bool = True) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, perforation_mode=perforation_mode[0],
                             grad_conv=grad_conv)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, perforation_mode=perforation_mode[1],
                             grad_conv=grad_conv)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, perforation_mode=perforation_mode[2],
                             grad_conv=grad_conv)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            perforation_mode: list = None,
            grad_conv: bool = True, extra_name="", in_size=(1, 3, 32, 32)
    ) -> None:
        super().__init__()
        self.extra_name = extra_name
        self.perforation = perforation_mode
        self.grad_conv = grad_conv
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        conv_in_block = (2 if block == BasicBlock else 3 if block == Bottleneck else -1)
        h = 1 if (self.inplanes != 64 * block.expansion) else 0
        n = sum(layers) * conv_in_block + 4 + h
        if type(self.perforation) == tuple:
            self.perforation = [self.perforation] * n
        elif type(self.perforation) not in [list, np.ndarray]:
            raise NotImplementedError("Provide the perforation mode")
        if n != len(self.perforation):
            raise ValueError(
                f"The perforation list length should equal the number of conv layers ({n})")
        self.conv1 = PerforatedConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,
                                      perforation_mode=self.perforation[0],
                                      grad_conv=grad_conv)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       perforation_mode=self.perforation[1:(layers[0] * conv_in_block) + 1 + h],
                                       grad_conv=grad_conv)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       perforation_mode=self.perforation[(layers[0] * conv_in_block) + 1 + h:
                                                                         (layers[1] + layers[0]) * conv_in_block + 2 + h],
                                       grad_conv=grad_conv)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       perforation_mode=
                                       self.perforation[(layers[1] + layers[0]) * conv_in_block + 2 + h:
                                                        (layers[2] + layers[1] + layers[0]) * conv_in_block + 3 + h],
                                       grad_conv=grad_conv)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       perforation_mode=
                                       self.perforation[(layers[2] + layers[1] + layers[0]) * conv_in_block + 3 + h:
                                                        (layers[3] + layers[2] + layers[1] + layers[
                                                            0]) * conv_in_block + 4 + h],
                                       grad_conv=grad_conv)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.in_size = in_size
        for m in self.modules():
            if isinstance(m, PerforatedConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        #init the net for perf sizes (unneeded, but if you want to know the net parameters before first batch it is necessary)
        self._reset()
    def _reset(self):
        self.eval()
        self(torch.zeros(self.in_size, device=self.conv1.weight.device))
        self.train()


    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False, perforation_mode: list = None,
            grad_conv: bool = True
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        conv_in_block = (2 if block == BasicBlock else 3 if block == Bottleneck else -1)
        downs = (1 if stride != 1 or self.inplanes != planes * block.expansion else 0)
        if type(self.perforation) == tuple:
            perforation_mode = [perforation_mode] * (blocks * conv_in_block + downs)
        if blocks * conv_in_block + downs != len(perforation_mode):
            raise ValueError(
                f"The perforation list length should equal the number of conv layers ({blocks * conv_in_block + downs}), in _make_layer()")
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, perforation_mode=perforation_mode[0],
                        grad_conv=grad_conv),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
                perforation_mode[downs:downs + conv_in_block], grad_conv=grad_conv
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    perforation_mode=perforation_mode[downs + conv_in_block * i:downs + conv_in_block * (i + 1)],
                    grad_conv=grad_conv
                )
            )

        return nn.Sequential(*layers)

    def _set_perforation(self, perf):
        if type(perf) == tuple:
            perf = [perf] * len(self._get_n_calc())
        self.perforation = perf
        cnt = 1
        self.conv1.perf_stride = perf[0]
        self.conv1.recompute = True
        ls = [self.layer1, self.layer2, self.layer3, self.layer4]
        for l in ls:
            if type(l) == BasicBlock:
                l.conv1.perf_stride = perf[cnt]
                l.conv1.recompute = True
                cnt += 1
                l.conv2.perf_stride = perf[cnt]
                l.conv2.recompute = True
                cnt += 1
                if l.downsample is not None:
                    l.downsample[0].perf_stride = perf[cnt]
                    l.downsample[0].recompute = True
                    cnt += 1
            elif type(l) == Bottleneck:
                l.conv1.perf_stride = perf[cnt]
                l.conv1.recompute = True
                cnt += 1
                l.conv2.perf_stride = perf[cnt]
                l.conv2.recompute = True
                cnt += 1
                l.conv3.perf_stride = perf[cnt]
                l.conv3.recompute = True
                cnt += 1
                if l.downsample is not None:
                    l.downsample[0].perf_stride = perf[cnt]
                    l.downsample[0].recompute = True
                    cnt += 1
            elif type(l) == Sequential:
                for ll in l:
                    if type(ll) == BasicBlock:
                        ll.conv1.perf_stride = perf[cnt]
                        ll.conv1.recompute = True
                        cnt += 1
                        ll.conv2.perf_stride = perf[cnt]
                        ll.conv2.recompute = True
                        cnt += 1
                        if ll.downsample is not None:
                            ll.downsample[0].perf_stride = perf[cnt]
                            ll.downsample[0].recompute = True
                            cnt += 1
                    elif type(ll) == Bottleneck:
                        ll.conv1.perf_stride = perf[cnt]
                        ll.conv1.recompute = True
                        cnt += 1
                        ll.conv2.perf_stride = perf[cnt]
                        ll.conv2.recompute = True
                        cnt += 1
                        ll.conv3.perf_stride = perf[cnt]
                        ll.conv3.recompute = True
                        cnt += 1
                        if ll.downsample is not None:
                            ll.downsample[0].perf_stride = perf[cnt]
                            ll.downsample[0].recompute = True
                            cnt += 1
        

    def _get_perforation(self):
        perfs = [self.conv1.perf_stride]
        ls = [self.layer1, self.layer2, self.layer3, self.layer4]
        for l in ls:
            if type(l) == BasicBlock:
                perfs.append(l.conv1.perf_stride)
                perfs.append(l.conv2.perf_stride)
                if l.downsample is not None:
                    perfs.append(l.downsample[0].perf_stride)
            elif type(l) == Bottleneck:
                perfs.append(l.conv1.perf_stride)
                perfs.append(l.conv2.perf_stride)
                perfs.append(l.conv3.perf_stride)
                if l.downsample is not None:
                    perfs.append(l.downsample[0].perf_stride)
            elif type(l) == Sequential:
                for ll in l:
                    if type(ll) == BasicBlock:
                        perfs.append(ll.conv1.perf_stride)
                        perfs.append(ll.conv2.perf_stride)
                        if ll.downsample is not None:
                            perfs.append(ll.downsample[0].perf_stride)
                    elif type(ll) == Bottleneck:
                        perfs.append(ll.conv1.perf_stride)
                        perfs.append(ll.conv2.perf_stride)
                        perfs.append(ll.conv3.perf_stride)
                        if ll.downsample is not None:
                            perfs.append(ll.downsample[0].perf_stride)
        self.perforation = perfs
        return perfs
    def _get_n_calc(self):
        perfs = [self.conv1.calculations]
        ls = [self.layer1, self.layer2, self.layer3, self.layer4]
        for l in ls:
            if type(l) == BasicBlock:
                perfs.append(l.conv1.calculations)
                perfs.append(l.conv2.calculations)
                if l.downsample is not None:
                    perfs.append(l.downsample[0].calculations)
            elif type(l) == Bottleneck:
                perfs.append(l.conv1.calculations)
                perfs.append(l.conv2.calculations)
                perfs.append(l.conv3.calculations)
                if l.downsample is not None:
                    perfs.append(l.downsample[0].calculations)
            elif type(l) == Sequential:
                for ll in l:
                    if type(ll) == BasicBlock:
                        perfs.append(ll.conv1.calculations)
                        perfs.append(ll.conv2.calculations)
                        if ll.downsample is not None:
                            perfs.append(ll.downsample[0].calculations)
                    elif type(ll) == Bottleneck:
                        perfs.append(ll.conv1.calculations)
                        perfs.append(ll.conv2.calculations)
                        perfs.append(ll.conv3.calculations)
                        if ll.downsample is not None:
                            perfs.append(ll.downsample[0].calculations)
        return perfs

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor, ) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


def resnet18(*, weights: Optional = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


def resnet34(*, weights: Optional = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet34_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


def resnet50(*, weights: Optional = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def resnet101(*, weights: Optional = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet101_Weights
        :members:
    """

    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def resnet152(*, weights: Optional = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet152_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet152_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet152_Weights
        :members:
    """

    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)


def resnext50_32x4d(
        *, weights: Optional = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:
    """

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def resnext101_32x8d(
        *, weights: Optional = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-101 32x8d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_32X8D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_32X8D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_32X8D_Weights
        :members:
    """

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def resnext101_64x4d(
        *, weights: Optional = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-101 64x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_64X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_64X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_64X4D_Weights
        :members:
    """

    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def wide_resnet50_2(
        *, weights: Optional = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Wide ResNet-50-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet50_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet50_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet50_2_Weights
        :members:
    """

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def wide_resnet101_2(
        *, weights: Optional = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Wide ResNet-101-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048
    channels, and in Wide ResNet-101-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet101_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet101_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet101_2_Weights
        :members:
    """

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)
