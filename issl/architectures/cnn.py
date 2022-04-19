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

__all__ = ["ResNet", "ResNetTranspose",  "ConvNext"]


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
        is_channel_out_dim: bool=False,
        bottleneck_channel: Optional[int] = None,
        is_bn_bttle_channel: bool = False,  # TODO remove after chose better
        bottleneck_mode : str = "linear"  # linear / cnn / mlp
    ):
        super().__init__()
        kwargs = {}
        self.in_shape = in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.out_dim = prod(self.out_shape)
        self.is_pretrained = is_pretrained
        self.is_no_linear = is_no_linear
        self.is_channel_out_dim = is_channel_out_dim
        self.bottleneck_channel = bottleneck_channel
        self.is_bn_bttle_channel = is_bn_bttle_channel
        self.bottleneck_mode = bottleneck_mode

        if not self.is_pretrained:
            # cannot load pretrained if wrong out dim
            kwargs["num_classes"] = self.out_dim

        self.resnet = torchvision.models.__dict__[base](
            pretrained=self.is_pretrained,
            **kwargs,
        )

        if self.is_channel_out_dim:
            self.update_out_chan_()

        if self.is_no_linear or self.is_pretrained or self.is_channel_out_dim:
            # when pretrained has to remove last layer
            self.rm_last_linear_()

        self.update_conv_size_(self.in_shape)

        self.reset_parameters()

    def update_out_chan_(self):
        if self.bottleneck_channel is None:
            conv1 = nn.Conv2d(self.resnet.fc.in_features, self.out_dim, kernel_size=1, bias=False)

        else:  # TODO chose best
            if self.bottleneck_mode == "linear":
                # low rank linear
                conv1_to_bttle = nn.Conv2d(self.resnet.fc.in_features, self.bottleneck_channel, kernel_size=1, bias=False)
                conv1_from_bttle = nn.Conv2d(self.bottleneck_channel, self.out_dim, kernel_size=1, bias=False)

                if self.is_bn_bttle_channel:
                    bn = torch.nn.BatchNorm2d(self.bottleneck_channel, affine=False)
                    conv1 = nn.Sequential(conv1_to_bttle, bn, conv1_from_bttle)
                else:
                    conv1 = nn.Sequential(conv1_to_bttle, conv1_from_bttle)

            elif self.bottleneck_mode == "mlp":
                conv1_to_bttle = nn.Conv2d(self.resnet.fc.in_features,
                                           self.bottleneck_channel,
                                           kernel_size=1,
                                           bias=False) # will use batchnorm
                conv1_from_bttle = nn.Conv2d(self.bottleneck_channel, self.out_dim,
                                             kernel_size=1, bias=False)
                bn = torch.nn.BatchNorm2d(self.bottleneck_channel)
                conv1 = nn.Sequential(conv1_to_bttle, bn, nn.ReLU(), conv1_from_bttle)

            elif self.bottleneck_mode == "cnn":
                # use depth wise seprable convolutions to be more efficient parameter wise
                depthconv1_to_bttle = nn.Conv2d(self.resnet.fc.in_features,
                                                self.resnet.fc.in_features,
                                                groups=self.resnet.fc.in_features,
                                                stride=1,  padding=1, kernel_size=3,
                                                bias=False)  # will use batchnorm
                pointconv1_to_bttle = nn.Conv2d(self.resnet.fc.in_features,
                                               self.bottleneck_channel,
                                               kernel_size=1, bias=False)  # will use batchnorm

                bn = torch.nn.BatchNorm2d(self.bottleneck_channel)

                depthconv1_from_bttle = nn.Conv2d(self.bottleneck_channel,
                                                self.bottleneck_channel,
                                                groups=self.bottleneck_channel,
                                                stride=1, padding=1, kernel_size=3,
                                                bias=False)  # will use batchnorm
                pointconv1_from_bttle = nn.Conv2d(self.bottleneck_channel,
                                                self.out_dim,
                                                kernel_size=1, bias=False)  # will use batchnorm

                conv1 = nn.Sequential(depthconv1_to_bttle, pointconv1_to_bttle,
                                      bn, nn.ReLU(),
                                      depthconv1_from_bttle, pointconv1_from_bttle)

            else:
                raise ValueError(f"Unknown self.bottleneck_mode={self.bottleneck_mode}.")

        bn = torch.nn.BatchNorm2d(self.out_dim)

        resizer = nn.Sequential(conv1, bn, torch.nn.ReLU(inplace=True))

        weights_init(resizer)

        self.resnet.avgpool = nn.Sequential(resizer, self.resnet.avgpool)

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

    def forward(self, X, rm_out_chan=False):
        if rm_out_chan:  # TODO test if works and worth keeping
            assert self.is_channel_out_dim
            breakpoint()
            old_avg_pool = self.resnet.avgpool
            self.resnet.avgpool = self.resnet.avgpool[1]
            Y_pred = self.resnet(X)
            self.resnet.avgpool = old_avg_pool

        else:
            Y_pred = self.resnet(X)
            Y_pred = Y_pred.unflatten(dim=-1, sizes=self.out_shape)
        return Y_pred

    def reset_parameters(self):
        # resnet is already correctly initialized
        pass

class ConvNext(ResNet):
    def update_out_chan_(self):
        #TODO chose whether or not to use norm
        old_out_dim = self.resnet.classifier[2].in_features
        #norm = torchvision.models.convnext.LayerNorm2d(old_out_dim, eps=1e-6)
        conv1 = nn.Conv2d(old_out_dim, self.out_dim, kernel_size=1)
        nn.init.trunc_normal_(conv1.weight, std=0.02)  # not adding GELU because will directly be followed by layernorm
        #resizer = nn.Sequential(norm, conv1)
        resizer = conv1
        self.resnet.avgpool = nn.Sequential(resizer, self.resnet.avgpool)
        self.resnet.classifier[0] = torchvision.models.convnext.LayerNorm2d(self.out_dim, eps=1e-6)

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


