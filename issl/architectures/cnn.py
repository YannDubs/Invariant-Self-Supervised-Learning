from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any, Optional

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as transform_lib

from issl.architectures.helpers import (
    closest_pow,
    get_Activation,
    get_Normalization,
    is_pow2,
)
from issl.helpers import prod, weights_init

logger = logging.getLogger(__name__)

__all__ = ["ResNet", "CNN", "CNNUnflatten"]


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
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        base: str = "resnet18",
        is_pretrained: bool = False,
        norm_layer: str = "batchnorm",
    ):
        super().__init__()
        kwargs = {}
        self.in_shape = in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.out_dim = prod(self.out_shape)
        self.is_pretrained = is_pretrained

        if not self.is_pretrained:
            # cannot load pretrained if wrong out dim
            kwargs["num_classes"] = self.out_dim

        self.resnet = torchvision.models.__dict__[base](
            pretrained=self.is_pretrained,
            norm_layer=get_Normalization(norm_layer, 2),
            **kwargs,
        )

        if self.is_pretrained:
            assert self.out_dim == self.resnet.fc.in_features
            # when pretrained has to remove last layer
            self.resnet.fc = torch.nn.Identity()

        if self.in_shape[1] < 100:
            # resnet for smaller images
            self.resnet.conv1 = nn.Conv2d(
                in_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.resnet.maxpool = nn.Identity()

        self.reset_parameters()

    def forward(self, X):
        Y_pred = self.resnet(X)
        Y_pred = Y_pred.unflatten(dim=-1, sizes=self.out_shape)
        return Y_pred

    def reset_parameters(self):
        # resnet is already correctly initialized
        if self.in_shape[1] < 100:
            weights_init(self.resnet.conv1)


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

    out_dim : int
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
        out_dim: int,
        hid_dim: int = 32,
        norm_layer: str = "batchnorm",
        activation: str = "ReLU",
        n_layers: Optional[int] = None,
        **kwargs,
    ) -> None:

        super().__init__()

        in_shape, out_dim, resizer = self.validate_sizes(out_dim, in_shape)

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
