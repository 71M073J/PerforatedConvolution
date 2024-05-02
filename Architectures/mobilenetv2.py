from functools import partial
from typing import Any, Callable, List, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn import Sequential

from .conv2dNormActivation import Conv2dNormActivation
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
from .PerforatedConv2d import PerforatedConv2d

__all__ = ["MobileNetV2", "MobileNet_V2_Weights", "mobilenet_v2"]


# necessary for backwards compatibility
class InvertedResidual(nn.Module):
    def __init__(
            self, inp: int, oup: int, stride: int, expand_ratio: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            perforation_mode: list = None,
            grad_conv: bool = True
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6,
                                     perforation_mode=perforation_mode[-3], grad_conv=grad_conv)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                    perforation_mode=perforation_mode[-2],
                    grad_conv=grad_conv
                ),
                # pw-linear
                PerforatedConv2d(hidden_dim, oup, 1, 1, 0, bias=False, perforation_mode=perforation_mode[-1],
                                 grad_conv=grad_conv),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            inverted_residual_setting: Optional[List[List[int]]] = None,
            round_nearest: int = 8,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.2,
            perforation_mode: Union[tuple, list] = None,
            grad_conv: bool = True, extra_name="", in_size=(1, 3, 32, 32)
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        super().__init__()
        _log_api_usage_once(self)
        self.grad_conv = grad_conv

        self.extra_name = extra_name
        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        self.perforation = perforation_mode
        if type(self.perforation) == tuple:
            self.perforation = [self.perforation] * (
                        sum([x[2] * (2 if x[0] == 1 else 3) for x in inverted_residual_setting]) + 2)
        elif type(self.perforation) != list:
            raise NotImplementedError("Provide the perforation mode")

        if sum([x[2] * (2 if x[0] == 1 else 3) for x in inverted_residual_setting]) + 2 != len(self.perforation):
            raise ValueError(
                f"The perforation list length should equal the number of conv layers, {sum([x[2] * (2 if x[0] == 1 else 3) for x in inverted_residual_setting]) + 2}")
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6,
                                 perforation_mode=self.perforation[0], grad_conv=grad_conv)
        ]
        # building inverted residual blocks
        cnt = 1
        for ind, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            n_conv_in_block = (2 if t == 1 else 3)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer,
                                      perforation_mode=self.perforation[cnt:cnt + n_conv_in_block],
                                      grad_conv=grad_conv))
                cnt += n_conv_in_block
                input_channel = output_channel

        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6,
                perforation_mode=self.perforation[-1], grad_conv=grad_conv
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
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
        self(torch.zeros(self.in_size, device=self.features[0].weight.device))
        self.train()
        # init the net for perf sizes (unneeded, but if you want to know the net parameters before first batch it is necessary)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
                for c in layer.conv:
                    if type(c) == Conv2dNormActivation:
                        c[0].perf_stride = perf[cnt]
                        c[0].recompute = True
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
                            # elif type(cc) ==
                            cnt += 1

        self._reset()

    def _get_perforation(self):
        perfs = []
        for layer in self.features:
            if type(layer) == Conv2dNormActivation:
                perfs.append(layer[0].perf_stride)
            elif type(layer) == InvertedResidual:
                for c in layer.conv:
                    if type(c) == Conv2dNormActivation:
                        perfs.append(c[0].perf_stride)
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
                for c in layer.conv:
                    if type(c) == Conv2dNormActivation:
                        perfs.append(c[0].calculations)
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
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v2(
        *, weights: Optional = None, progress: bool = True, **kwargs: Any
) -> MobileNetV2:
    """MobileNetV2 architecture from the `MobileNetV2: Inverted Residuals and Linear
    Bottlenecks <https://arxiv.org/abs/1801.04381>`_ paper.

    Args:
        weights (:class:`~torchvision.models.MobileNet_V2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileNet_V2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mobilenetv2.MobileNetV2``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MobileNet_V2_Weights
        :members:
    """

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV2(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model
