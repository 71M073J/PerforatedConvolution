from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import Sequential

#from torchvision.ops.misc import SqueezeExcitation as SElayer
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _make_divisible, _ovewrite_named_param
from .conv2dNormActivation import Conv2dNormActivation
from .PerforatedConv2d import PerforatedConv2d

class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid, perforation_mode: tuple = None,
                 grad_conv: bool = True,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = PerforatedConv2d(input_channels, squeeze_channels, 1, perforation_mode=perforation_mode[0],
                                    grad_conv=grad_conv)
        self.fc2 = PerforatedConv2d(squeeze_channels, input_channels, 1, perforation_mode=perforation_mode[1],
                                    grad_conv=grad_conv)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input

class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
            self,
            input_channels: int,
            kernel: int,
            expanded_channels: int,
            out_channels: int,
            use_se: bool,
            activation: str,
            stride: int,
            dilation: int,
            width_mult: float, perforation_mode: list = None,
            grad_conv: bool = True
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation
        self.perforation_mode = perforation_mode
        self.grad_conv = grad_conv

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
            self,
            cnf: InvertedResidualConfig,
            norm_layer: Callable[..., nn.Module],
            se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        perforation_mode = cnf.perforation_mode
        grad_conv = cnf.grad_conv
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        eq_channels = cnf.expanded_channels != cnf.input_channels
        # expand
        if eq_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer, perforation_mode=perforation_mode[0]
                    ,grad_conv=grad_conv
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                perforation_mode=perforation_mode[1 if eq_channels else 0],
                grad_conv=grad_conv
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels,
                                   perforation_mode=perforation_mode[2 if eq_channels else 1:4 if eq_channels else 3],
                                   grad_conv=grad_conv))

        # project
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None,
                perforation_mode=perforation_mode[4 if (eq_channels and cnf.use_se) else (3 if cnf.use_se else 1)],
                grad_conv=grad_conv
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.2, perforation_mode: list = None,
            grad_conv: bool = True, extra_name="", in_size=(1, 3, 32, 32),
            **kwargs: Any,
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """

        self.extra_name=extra_name
        self.size = kwargs["size"]
        super().__init__()
        self.perforation = perforation_mode
        self.grad_conv = grad_conv
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        self.perforation = perforation_mode
        if type(self.perforation) == tuple:
            self.perforation = [self.perforation] * (sum([len(x.perforation_mode) for x in inverted_residual_setting]) + 2)
        if (sum([len(x.perforation_mode) for x in inverted_residual_setting]) + 2) != len(self.perforation):
            raise ValueError(f"The perforation list length should equal the number of conv layers, {(sum([len(x.perforation_mode) for x in inverted_residual_setting]) + 2)}")
        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
                perforation_mode=self.perforation[0],
                grad_conv=grad_conv
            )
        )

        # building inverted residual blocks
        for ii, cnf in enumerate(inverted_residual_setting):
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
                perforation_mode=self.perforation[-1],
                grad_conv=grad_conv

            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, PerforatedConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        self.in_size = in_size
        self._reset()

    def _reset(self):
        self.eval()
        self(torch.zeros(self.in_size, device=self.features[0][0].weight.device))
        self.train()
        #init the net for perf sizes (unneeded, but if you want to know the net parameters before first batch it is necessary)

    def _set_perforation(self, perf):

        if type(perf) == tuple:
            perf = [perf] * len(self._get_n_calc())
        self.perforation = perf

        cnt = 0
        for layer in self.features:
            if type(layer) == Conv2dNormActivation:
                layer[0].perf_stride = perf[cnt]
                layer[0].recompute = True
                cnt += 1
            elif type(layer) == InvertedResidual:
                for c in layer.block:
                    if type(c) == Conv2dNormActivation:
                        c[0].perf_stride = perf[cnt]
                        c[0].recompute = True
                        cnt += 1
                    elif type(c) == SqueezeExcitation:
                        c.fc1.perf_stride = perf[cnt]
                        c.fc1.recompute = True
                        cnt += 1
                        c.fc2.perf_stride = perf[cnt]
                        c.fc2.recompute = True
                        cnt += 1

                    elif type(c) == PerforatedConv2d:
                        c.perf_stride = perf[cnt]
                        c.recompute = True
                        cnt += 1
            elif type(layer) == Sequential:
                for c in layer:
                    if type(c) == Conv2dNormActivation:
                        c[0].perf_stride = perf[cnt]
                        c[0].recompute = True
                        cnt += 1
                    if type(c) == InvertedResidual:
                        for cc in layer[0].conv:
                            if type(cc) == Conv2dNormActivation:
                                cc[0].perf_stride = perf[cnt]
                                cc[0].recompute = True
                                cnt += 1

        

    def _get_perforation(self):
        perfs = []
        for layer in self.features:
            if type(layer) == Conv2dNormActivation:
                perfs.append(layer[0].perf_stride)
            elif type(layer) == InvertedResidual:
                for c in layer.block:
                    if type(c) == Conv2dNormActivation:
                        perfs.append(c[0].perf_stride)
                    elif type(c) == SqueezeExcitation:
                        perfs.append(c.fc1.perf_stride)
                        perfs.append(c.fc2.perf_stride)
                    elif type(c) == PerforatedConv2d:
                        perfs.append(c.perf_stride)
            elif type(layer) == Sequential:
                for c in layer:
                    if type(c) == Conv2dNormActivation:
                        perfs.append(c[0].perf_stride)
                    if type(c) == InvertedResidual:
                        for cc in layer[0].conv:
                            if type(cc) == Conv2dNormActivation:
                                perfs.append(cc[0].perf_stride)

        self.perforation = perfs
        return perfs

    def _get_n_calc(self):
        perfs = []
        for layer in self.features:
            if type(layer) == Conv2dNormActivation:
                perfs.append(layer[0].calculations)
            elif type(layer) == InvertedResidual:
                for c in layer.block:
                    if type(c) == Conv2dNormActivation:
                        perfs.append(c[0].calculations)
                    elif type(c) == SqueezeExcitation:
                        perfs.append(c.fc1.calculations)
                        perfs.append(c.fc2.calculations)
                    elif type(c) == PerforatedConv2d:
                        perfs.append(c.calculations)
            elif type(layer) == Sequential:
                for c in layer:
                    if type(c) == Conv2dNormActivation:
                        perfs.append(c[0].calculations)
                    if type(c) == InvertedResidual:
                        for cc in layer[0].conv:
                            if type(cc) == Conv2dNormActivation:
                                perfs.append(cc[0].calculations)

        return perfs
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mobilenet_v3_conf(
        arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False,
        perforation_mode: list = None,
                 grad_conv: bool = True, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":

        if type(perforation_mode) == tuple:
            perforation_mode = [perforation_mode] * (60 + 2)
        elif type(perforation_mode) != list:
            raise NotImplementedError("Provide the perforation mode")

        if 62 != len(perforation_mode):
            raise ValueError(f"The perforation list length should equal the number of conv layers (62)")
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1, perforation_mode=perforation_mode[1:3],grad_conv=grad_conv),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1, perforation_mode=perforation_mode[3:6],grad_conv=grad_conv),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1, perforation_mode=perforation_mode[6:9],grad_conv=grad_conv),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1, perforation_mode=perforation_mode[9:14],grad_conv=grad_conv),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1, perforation_mode=perforation_mode[14:19],grad_conv=grad_conv),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1, perforation_mode=perforation_mode[19:24], grad_conv=grad_conv),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1, perforation_mode=perforation_mode[24:27], grad_conv=grad_conv),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1, perforation_mode=perforation_mode[27:30], grad_conv=grad_conv),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1, perforation_mode=perforation_mode[30:33],grad_conv=grad_conv),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1, perforation_mode=perforation_mode[33:36], grad_conv=grad_conv),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1, perforation_mode=perforation_mode[36:41],grad_conv=grad_conv),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1, perforation_mode=perforation_mode[41:46],  grad_conv=grad_conv),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation, perforation_mode=perforation_mode[46:51],  grad_conv=grad_conv),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation, perforation_mode=perforation_mode[51:56],  grad_conv=grad_conv),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation, perforation_mode=perforation_mode[56:61],  grad_conv=grad_conv),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":

        if type(perforation_mode) == tuple:
            perforation_mode = [perforation_mode] * 52
        elif type(perforation_mode) != list:
            raise NotImplementedError("Provide the perforation mode")

        if 52 != len(perforation_mode):
            raise ValueError(f"The perforation list length should equal the number of conv layers (52)")
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1, perforation_mode=perforation_mode[1:5],  grad_conv=grad_conv),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1, perforation_mode=perforation_mode[5:8],  grad_conv=grad_conv),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1, perforation_mode=perforation_mode[8:11],  grad_conv=grad_conv),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1, perforation_mode=perforation_mode[11:16],   grad_conv=grad_conv),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1, perforation_mode=perforation_mode[16:21],  grad_conv=grad_conv),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1, perforation_mode=perforation_mode[21:26],  grad_conv=grad_conv),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1, perforation_mode=perforation_mode[26:31],  grad_conv=grad_conv),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1, perforation_mode=perforation_mode[31:36],  grad_conv=grad_conv),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation, perforation_mode=perforation_mode[36:41],  grad_conv=grad_conv),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation, perforation_mode=perforation_mode[41:46],  grad_conv=grad_conv),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation, perforation_mode=perforation_mode[46:51],  grad_conv=grad_conv),
        ]

        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel



def _mobilenet_v3(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MobileNetV3:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model



def mobilenet_v3_large(
    *, weights: Optional = None, progress: bool = True, **kwargs: Any
) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__.

    Args:
        weights (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mobilenet.MobileNetV3``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MobileNet_V3_Large_Weights
        :members:
    """

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, size=45,**kwargs)


def mobilenet_v3_small(
    *, weights: Optional= None, progress: bool = True, **kwargs: Any
) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__.

    Args:
        weights (:class:`~torchvision.models.MobileNet_V3_Small_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileNet_V3_Small_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mobilenet.MobileNetV3``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MobileNet_V3_Small_Weights
        :members:
    """

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, size=33, **kwargs)