from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn

from issl.architectures.basic import Resizer

try:
    # only if you want to use a ViT
    from transformers import ViTForImageClassification
    from transformers import AutoConfig
except ImportError:
    pass

SHORTHANDS_HUGGING = {
    "vit-B/16": "google/vit-base-patch16-224-in21k",
    "vit-B/32": "google/vit-base-patch32-224-in21k",
    "vit-H/14": "google/vit-huge-patch14-224-in21k",
    "vit-L/16": "google/vit-large-patch16-224-in21k",
    "deit-T/16": "facebook/deit-tiny-patch16-224",
    "deit-S/16": "facebook/deit-small-patch16-224",
    "deit-B/16": "facebook/deit-base-patch16-224",
}


class ViT(nn.Module):
    """Base Class for a visual transformer.

    Notes
    -----
    - Currently only works with square images.

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first).

    out_shape : int or tuple
        Size of the output.

    base : {'vit-B/16', 'vit-B/32', 'vit-H/14', 'vit-L/16', 'deit-T/16', 'deit-S/16', 'deit-B/16'
            }U{https://huggingface.co/models?filter=vit}, optional
        Base transformer to use, any model found in https://huggingface.co/models?filter=vit should work. We provide
        shorthands.

    is_pretrained : bool, optional
        Whether to load a model pretrained on imagenet.
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: int,
        base: Optional[str] = None,
        is_pretrained: bool = True,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.base = SHORTHANDS_HUGGING.get(base, base)
        self.is_pretrained = is_pretrained

        n_channels, height, width = in_shape
        assert height == width

        self.config = AutoConfig.from_pretrained(base)
        self.config.num_channels = n_channels
        self.config.image_size = width
        curr_out_dim = self.config.hidden_size

        self.resizer = Resizer(curr_out_dim, self.out_shape)

        self.vit = ViTForImageClassification(self.config).vit
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.is_pretrained:
            self.vit = ViTForImageClassification.from_pretrained(self.base).vit
        else:
            self.vit = ViTForImageClassification(self.config).vit

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.vit(X)
        first_tok = out.last_hidden_state[:, 0, :]
        Y_pred = self.resizer(first_tok)
        return Y_pred
