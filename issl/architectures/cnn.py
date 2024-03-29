from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Optional
import einops
import math

import torchvision

import torch
import torch.nn as nn
from issl.helpers import prod, weights_init

logger = logging.getLogger(__name__)

__all__ = ["ResNet"]


class ResNet(nn.Module):
    """Base class for ResNets.

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). This is used to see whether to change the underlying
        resnet or not. If first dim < 100, then will decrease the kernel size  and stride of the
        first conv, and remove the max pooling layer as done (for cifar10) in
        https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469.

    out_dim : int
        Size of the output.

    base : {'resnet18', 'resnet34', 'resnet50', ...}, optional
        Base resnet to use, any model `torchvision.models.resnet` should work (the larger models were
        not tested).

    norm_layer : nn.Module or {"identity","batch"}, optional
        Normalizing layer to use.

    is_channel_out_dim : bool, optional
        Whether to change the dimension of the output using the channels before the pooling layer
        rather than a linear mapping after the pooling.

    kwargs: dict, optional
        Additional arguments to pass to the resnet constructor.
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_dim: int,
        base: str = "resnet18",
        is_channel_out_dim: bool=False,
        bottleneck_channel: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.in_shape = in_shape
        self.out_dim = out_dim
        self.is_channel_out_dim = is_channel_out_dim
        self.bottleneck_channel = bottleneck_channel

        self.resnet = torchvision.models.__dict__[base](**kwargs)

        if self.is_channel_out_dim:
            self.update_out_chan_()

        # remove last linear layer in SSL
        self.rm_last_linear_()

        self.update_conv_size_(self.in_shape)

        self.reset_parameters()

    def update_out_chan_(self):
        current_nchan = self.resnet.fc.in_features
        target_nchan = self.out_dim

        if self.bottleneck_channel is None:
            conv1 = nn.Conv2d(current_nchan, target_nchan, kernel_size=1, bias=False)
            bn = torch.nn.BatchNorm2d(target_nchan)
            weights_init(bn)
            resizer = nn.Sequential(conv1, bn, torch.nn.ReLU(inplace=True))

        else:

            if target_nchan % current_nchan == 0:
                resizer = BottleneckExpand(current_nchan,
                                           self.bottleneck_channel,
                                           expansion=target_nchan // current_nchan)
            elif target_nchan < current_nchan:
                resizer = BottleneckExpand(current_nchan,
                                           self.bottleneck_channel,
                                           is_residual=False,
                                           expansion=target_nchan / current_nchan)
            else:
                raise ValueError(f"Cannot deal with target_nchan={target_nchan} and current_nchan={current_nchan}.")

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
            # following https://github.com/htdt/self-supervised/blob/d24f3c722ac4e945161a9cd8c830bf912403a8d7/model.py#L19
            # this should only be removed for cifar
            self.resnet.maxpool = nn.Identity()

    def forward(self, X):
        Y_pred = self.resnet(X)
        return Y_pred

    def reset_parameters(self):
        # resnet is already correctly initialized
        pass


class BottleneckExpand(nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_channels,
            expansion=8,
            norm_layer=None,
            is_residual=True
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.expansion = expansion
        self.is_residual = is_residual
        out_channels = int(in_channels * self.expansion)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, bias=False, padding=1)
        self.bn2 = norm_layer(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.reset_parameters()

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

        if self.is_residual:
            identity = einops.repeat(identity, 'b c h w -> b (tile c) h w', tile=self.expansion)
            out += identity

        out = self.relu(out)

        return out

    def reset_parameters(self) -> None:
        weights_init(self)
        # using Johnson-Lindenstrauss lemma for initialization of the projection matrix
        torch.nn.init.normal_(self.conv1.weight,
                              std=1 / math.sqrt(self.conv1.weight.shape[0]))

