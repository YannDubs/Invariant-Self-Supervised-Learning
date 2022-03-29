from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any, Optional, Union

import torchvision
from torchvision import transforms as transform_lib

import torch
import torch.nn as nn
import torch.nn.functional as F
from issl.architectures.helpers import (
    closest_pow,
    get_Activation,
    get_Normalization,
    is_pow2,
)
from issl.helpers import check_import, prod, weights_init

logger = logging.getLogger(__name__)

__all__ = ["ResNet", "ResNetTranspose", "CNN", "CNNUnflatten", "ConvNext"]


class ResNet(nn.Module):
    """Base class for ResNets.

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). This is used to see whether to change the underlying
        resnet or not. If first dim < 100, then will decrease the kernel size  and stride of the
        first conv, and remove the max pooling layer as done (for cifar10) in
        https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469.

    out_shape : int or tuple
        Size of the output.

    base : {'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2'}, optional
        Base resnet to use, any model `torchvision.models.resnet` should work (the larger models were
        not tested).

    is_pretrained : bool, optional
        Whether to load a model pretrained on imagenet. Might not work well with `is_small=True`.

    norm_layer : nn.Module or {"identity","batch"}, optional
        Normalizing layer to use.

    is_no_linear : bool, optional
        Whether or not to remove the last linear layer. This is typical in self supervised learning but
        has the disadvantage that you cannot chose the dimensionality of Z.

    is_channel_out_dim : bool, optional
        Whether to change the dimension of the output using the channels before the pooling layer
        rather than a linear mapping after the pooling.
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        base: str = "resnet18",
        is_pretrained: bool = False,
        is_no_linear: bool= False,
        is_channel_out_dim: bool=False
    ):
        super().__init__()
        kwargs = {}
        self.in_shape = in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.out_dim = prod(self.out_shape)
        self.is_pretrained = is_pretrained
        self.is_no_linear = is_no_linear
        self.is_channel_out_dim = is_channel_out_dim

        if not self.is_pretrained:
            # cannot load pretrained if wrong out dim
            kwargs["num_classes"] = self.out_dim

        self.resnet = torchvision.models.__dict__[base](
            pretrained=self.is_pretrained,
            **kwargs,
        )

        if self.is_channel_out_dim:
            # TODO 512 only works for resnet18. Make it work for convnext and others!
            conv1 = nn.Conv2d(512, self.out_dim, kernel_size=1, bias=False)
            bn = torch.nn.BatchNorm2d(self.out_dim)

            nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)

            resizer = nn.Sequential(conv1, bn, torch.nn.ReLU(inplace=True))

            self.resnet.avgpool = nn.Sequential(resizer, self.resnet.avgpool)

        if self.is_no_linear or self.is_pretrained or self.is_channel_out_dim:
            # when pretrained has to remove last layer
            self.rm_last_linear_()

        self.update_conv_size_(self.in_shape)

        self.reset_parameters()

    def rm_last_linear_(self):
        self.resnet.fc = nn.Identity()

    def update_conv_size_(self, in_shape):
        """Update network based on iamge size."""

        if in_shape[1] < 100:
            # resnet for smaller images
            self.resnet.conv1 = nn.Conv2d(
                in_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            weights_init(self.resnet.conv1)

        if in_shape[1] < 50:
            # following https://github.com/htdt/self-supervised/blob/d24f3c722ac4e945161a9cd8c830bf912403a8d7/model
            # .py#L19
            # this should only be removed for cifar
            self.resnet.maxpool = nn.Identity()

    def forward(self, X):
        Y_pred = self.resnet(X)
        Y_pred = Y_pred.unflatten(dim=-1, sizes=self.out_shape)
        return Y_pred

    def reset_parameters(self):
        # resnet is already correctly initialized
        pass

class ConvNext(ResNet):
    def rm_last_linear_(self):
        self.resnet.classifier[2] = nn.Identity()

    def update_conv_size_(self, in_shape):
        """Update network based on image size."""
        if in_shape[1] < 100:
            # convnext for smaller images (stride 4 -> 2)
            conv1 = self.resnet.features[0][0]
            self.resnet.features[0][0] = nn.Conv2d(
                in_shape[0], conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=2
            )
            weights_init(self.resnet.features[0][0])

class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def resize_conv3x3(in_planes, out_planes, scale=1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv3x3(in_planes, out_planes)
    else:
        return nn.Sequential(
            Interpolate(scale_factor=scale), conv3x3(in_planes, out_planes)
        )


def resize_conv1x1(in_planes, out_planes, scale=1):
    """upsample + 1x1 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv1x1(in_planes, out_planes)
    else:
        return nn.Sequential(
            Interpolate(scale_factor=scale), conv1x1(in_planes, out_planes)
        )


class DecoderBlock(nn.Module):
    """ResNet block, but convs replaced with resize convs, and channel increase is in second conv, not first."""

    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None, activation="ReLU"):
        super().__init__()
        self.conv1 = resize_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = get_Activation(activation)()
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecoderBottleneck(nn.Module):
    """ResNet bottleneck, but convs replaced with resize convs."""

    expansion = 4

    def __init__(self, inplanes, planes, scale=1, upsample=None, activation="ReLU"):
        super().__init__()
        width = planes  # this needs to change if we want wide resnets
        self.conv1 = resize_conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = resize_conv3x3(width, width, scale)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = get_Activation(activation)()
        self.upsample = upsample
        self.scale = scale

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetDecoder(nn.Module):
    """Resnet in reverse order."""

    def __init__(
        self,
        block,
        layers,
        latent_dim,
        input_height,
        first_conv=False,
        maxpool1=False,
        activation="ReLU",
    ):
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 512 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height
        self.activation = activation

        self.upscale_factor = 8

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 64, layers[2], scale=2)

        if self.maxpool1:
            self.layer4 = self._make_layer(block, 64, layers[3], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer4 = self._make_layer(block, 64, layers[3])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(
            64 * block.expansion, 3, kernel_size=3, stride=1, padding=1, bias=False
        )

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, scale, upsample, activation=self.activation)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=self.activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        x = x.view(x.size(0), 512 * self.expansion, 4, 4)
        x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x


class ResNetTranspose(nn.Module):
    # transposed version of the resnet. Can be used as a decoder.
    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        base: str = "resnet18",
        activation: str = "ReLU",
    ) -> None:
        super().__init__()

        self.out_shape = out_shape
        self.in_shape = [in_shape] if isinstance(in_shape, int) else in_shape
        self.in_dim = prod(self.in_shape)
        self.base = base
        self.activation = activation

        if self.base == "resnet18":
            block = DecoderBlock
            layers = [2, 2, 2, 2]
        elif self.base == "resnet34":
            block = DecoderBlock
            layers = [3, 4, 6, 3]
        elif self.base == "resnet50":
            block = DecoderBottleneck
            layers = [3, 4, 6, 3]
        elif self.base == "resnet101":
            block = DecoderBottleneck
            layers = [3, 4, 23, 3]
        else:
            raise ValueError(f"Unknown base = {self.base}")

        is_small = self.out_shape[1] < 100
        self.resnet = ResNetDecoder(
            block,
            layers,
            self.in_dim,
            self.out_shape[1],
            first_conv=not is_small,
            maxpool1=not is_small,
            activation=self.activation,
        )

        n_chan = self.out_shape[0]
        if n_chan != 3:
            # replace in case it's black and white
            self.resnet.conv1 = nn.Conv2d(
                self.resnet.conv1.in_channels,
                n_chan,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights_init(self)

    def forward(self, Z: torch.Tensor):
        Z = Z.flatten(start_dim=Z.ndim - len(self.in_shape))
        X_hat = self.resnet(Z)
        return X_hat


class CNN(nn.Module):
    """CNN in shape of pyramid, which doubles hidden after each layer but decreases image size by 2.

    Notes
    -----
    - if some of the sides of the inputs are not power of 2 they will be resized to the closest power
    of 2 for prediction.
    - If `in_shape` and `out_dim` are reversed (i.e. `in_shape` is int) then will transpose the CNN.

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). If integer and `out_dim` is a tuple of int, then will
        transpose ("reverse") the CNN.

    out_shape : int
        Number of output channels. If tuple of int  and `in_shape` is an int, then will transpose
        ("reverse") the CNN.

    hid_dim : int, optional
        Base number of temporary channels (will be multiplied by 2 after each layer).

    norm_layer : callable or {"batchnorm", "identity"}
        Layer to return.

    activation : {any torch.nn activation}, optional
        Activation to use.

    n_layers : int, optional
        Number of layers. If `None` uses the required number of layers so that the smallest side
        is 2 after encoding (i.e. one less than the maximum).

    kwargs :
        Additional arguments to `ConvBlock`.
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Union[int, Sequence[int]],
        hid_dim: int = 32,
        norm_layer: str = "batchnorm",
        activation: str = "ReLU",
        n_layers: Optional[int] = None,
        **kwargs,
    ) -> None:

        super().__init__()

        in_shape, out_dim, resizer = self.validate_sizes(out_shape, in_shape)

        self.in_shape = in_shape
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.norm_layer = norm_layer
        self.activation = activation
        self.n_layers = n_layers

        if self.n_layers is None:
            # divide length by 2 at every step until smallest is 2
            min_side = min(self.in_shape[1], self.in_shape[2])
            self.n_layers = int(math.log2(min_side) - 1)

        Norm = get_Normalization(self.norm_layer, 2)
        # don't use bias with batch_norm https://twitter.com/karpathy/status/1013245864570073090?l...
        is_bias = Norm == nn.Identity

        # for size 32 will go 32,16,8,4,2
        # channels for hid_dim=32: 3,32,64,128,256
        channels = [self.in_shape[0]]
        channels += [self.hid_dim * (2 ** i) for i in range(0, self.n_layers)]
        end_h = self.in_shape[1] // (2 ** self.n_layers)
        end_w = self.in_shape[2] // (2 ** self.n_layers)

        if self.is_transpose:
            channels.reverse()

        layers = []
        in_chan = channels[0]
        for i, out_chan in enumerate(channels[1:]):
            is_last = i == len(channels[1:]) - 1
            layers += [
                self.make_block(in_chan, out_chan, Norm, is_bias, is_last, **kwargs)
            ]
            in_chan = out_chan

        if self.is_transpose:
            pre_layers = [
                nn.Linear(self.out_dim, channels[0] * end_w * end_h, bias=is_bias),
                nn.Unflatten(dim=-1, unflattened_size=(channels[0], end_h, end_w)),
            ]
            post_layers = [resizer]

        else:
            pre_layers = [resizer]
            post_layers = [
                nn.Flatten(start_dim=1),
                nn.Linear(channels[-1] * end_w * end_h, self.out_dim),
                # last layer should always have bias
            ]

        self.model = nn.Sequential(*(pre_layers + layers + post_layers))

        self.reset_parameters()

    def validate_sizes(
        self, out_dim: int, in_shape: Sequence[int]
    ) -> tuple[Sequence[int], int, Any]:
        if isinstance(out_dim, int) and not isinstance(in_shape, int):
            self.is_transpose = False
        else:
            in_shape, out_dim = out_dim, in_shape
            self.is_transpose = True

        resizer = nn.Identity()
        is_input_pow2 = is_pow2(in_shape[1]) and is_pow2(in_shape[2])
        if not is_input_pow2:
            # shape that you will work with which are power of 2
            in_shape_pow2 = list(in_shape)
            in_shape_pow2[1] = closest_pow(in_shape[1], base=2)
            in_shape_pow2[2] = closest_pow(in_shape[2], base=2)

            if self.is_transpose:
                # the model will output image of `in_shape_pow2` then will reshape to actual
                resizer = transform_lib.Resize((in_shape[1], in_shape[2]))
            else:
                # the model will first resize to power of 2
                resizer = transform_lib.Resize((in_shape_pow2[1], in_shape_pow2[2]))

            logger.warning(
                f"The input shape={in_shape} is not powers of 2 so we will rescale it and work with shape {in_shape_pow2}."
            )
            # for the rest treat the image as if pow 2
            in_shape = in_shape_pow2

        return in_shape, out_dim, resizer

    def make_block(
        self,
        in_chan: int,
        out_chan: int,
        Norm: Any,
        is_bias: bool,
        is_last: bool,
        **kwargs,
    ) -> nn.Module:

        if self.is_transpose:
            Activation = get_Activation(self.activation)
            block = [
                Norm(in_chan),
                Activation(in_chan),
                nn.ConvTranspose2d(
                    in_chan,
                    out_chan,
                    stride=2,
                    padding=1,
                    kernel_size=3,
                    output_padding=1,
                    bias=is_bias or is_last,
                    **kwargs,
                ),
            ]
        else:
            Activation = get_Activation(self.activation)
            block = [
                nn.Conv2d(
                    in_chan,
                    out_chan,
                    stride=2,
                    padding=1,
                    kernel_size=3,
                    bias=is_bias,
                    **kwargs,
                ),
                Norm(out_chan),
                Activation(out_chan),
            ]

        return nn.Sequential(*block)

    def reset_parameters(self) -> None:
        weights_init(self)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


class CNNUnflatten(nn.Module):
    """CNN where the output is an image, i.e., do not flatten output.

    Notes
    -----
    - replicates https://github.com/InterDigitalInc/CompressAI/blob/a73c3378e37a52a910afaf9477d985f86a06634d/compressai/models/priors.py#L104

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). If integer and `out_dim` is a tuple of int, then will
        transpose ("reverse") the CNN.

    out_dim : int
        Number of output channels. If tuple of int  and `in_shape` is an int, then will transpose
        ("reverse") the CNN.

    hid_dim : int, optional
        Number of channels for every layer.

    n_layers : int, optional
        Number of layers, after every layer divides image by 2 on each side.

    norm_layer : callable or {"batchnorm", "identity"}
        Normalization layer.

    activation : {any torch.nn activation}, optional
        Activation to use.
    """

    validate_sizes = CNN.validate_sizes

    def __init__(
        self,
        in_shape: Sequence[int],
        out_dim: int,
        hid_dim: int = 256,
        n_layers: int = 4,
        norm_layer: str = "batchnorm",
        activation: str = "ReLU",
    ):
        super().__init__()

        in_shape, out_dim, resizer = self.validate_sizes(out_dim, in_shape)

        self.in_shape = in_shape
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.activation = activation
        self.norm_layer = norm_layer

        # divide length by 2 at every step until smallest is 2
        end_h = self.in_shape[1] // (2 ** self.n_layers)
        end_w = self.in_shape[2] // (2 ** self.n_layers)

        # channels of the output latent image
        self.channel_out_dim = self.out_dim // (end_w * end_h)

        layers = [
            self.make_block(self.hid_dim, self.hid_dim)
            for _ in range(self.n_layers - 2)
        ]

        if self.is_transpose:
            pre_layers = [
                nn.Unflatten(
                    dim=-1, unflattened_size=(self.channel_out_dim, end_h, end_w)
                ),
                self.make_block(self.channel_out_dim, self.hid_dim),
            ]
            post_layers = [
                self.make_block(self.hid_dim, self.in_shape[0], is_last=True),
                resizer,
            ]

        else:
            pre_layers = [resizer, self.make_block(self.in_shape[0], self.hid_dim)]
            post_layers = [
                self.make_block(self.hid_dim, self.channel_out_dim, is_last=True),
                nn.Flatten(start_dim=1),
            ]

        self.model = nn.Sequential(*(pre_layers + layers + post_layers))

        self.reset_parameters()

    def make_block(
        self,
        in_chan: int,
        out_chan: int,
        is_last: bool = False,
        kernel_size: int = 5,
        stride: int = 2,
    ) -> nn.Module:
        if is_last:
            Norm = nn.Identity
        else:
            Norm = get_Normalization(self.norm_layer, 2)

        # don't use bias with batch_norm https://twitter.com/karpathy/status/1013245864570073090?l...
        is_bias = Norm == nn.Identity

        if self.is_transpose:
            conv = nn.ConvTranspose2d(
                in_chan,
                out_chan,
                kernel_size=kernel_size,
                stride=stride,
                output_padding=stride - 1,
                padding=kernel_size // 2,
                bias=is_bias,
            )
        else:
            conv = nn.Conv2d(
                in_chan,
                out_chan,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=is_bias,
            )

        if not is_last:
            Activation = get_Activation(self.activation)
            conv = nn.Sequential(conv, Norm(out_chan), Activation(out_chan))

        return conv

    def reset_parameters(self) -> None:
        weights_init(self)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)
