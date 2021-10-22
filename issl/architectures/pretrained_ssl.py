from __future__ import annotations

import torch
import torch.nn as nn

from issl.architectures.basic import Resizer
from issl.helpers import check_import

try:
    import clip
except ImportError:
    pass

try:
    from transformers import AutoModel
except ImportError:
    pass


try:
    from pl_bolts.models.self_supervised import SimCLR
    from pl_bolts.models.self_supervised.simclr.transforms import (
        SimCLRFinetuneTransform,
    )
    from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
    from pl_bolts.models.self_supervised import SwAV
    from pl_bolts.models.self_supervised.swav.transforms import SwAVFinetuneTransform
except ImportError:
    pass


class PretrainedSSL(nn.Module):
    """Pretrained self supervised models.

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). Needs to be 3,224,224.

    out_shape : int or tuple
        Size of the output. Flattened needs to be 512 for clip_vit, 1024 for clip_rn50, and
        2048 for swav and simclr.

    model : {"swav_rn50", "simclr_rn50", "clip_vitb16", "clip_vitb32", "clip_rn50", "dino_vitb16", "dino_rn50", "dino_vits16"}
        Which SSL model to use. "swav", "simclr", "clip_rn50", "dino_resnet50" are all resnet50 for fair comparison.
    """

    def __init__(self, in_shape, out_shape, model):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.model = model
        self.load_weights_()

        assert self.in_shape[0] == 3, "PretrainedSSL needs color images."
        assert self.in_shape[1] == self.in_shape[2] == 224

        if "clip_vit" in self.model:
            curr_out_dim = 512
        elif self.model == "clip_rn50":
            curr_out_dim = 1024
        elif self.model in ["swav_rn50", "simclr_rn50", "dino_rn50"]:
            curr_out_dim = 2048
        elif "dino_vitb" in self.model:
            curr_out_dim = 768
        elif "dino_vits" in self.model:
            curr_out_dim = 384
        else:
            raise ValueError(f"Unknown model={self.model}.")

        self.resizer = Resizer(curr_out_dim, self.out_shape)

        self.reset_parameters()

    def forward(self, X):
        z = self.encoder(X)
        z = self.resizer(z)
        return z

    def load_weights_(self):
        if self.model == "simclr_rn50":
            check_import("pl_bolts", "simclr in PretrainedSSL")
            # load resnet50 pretrained using SimCLR on imagenet
            weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
            self.encoder = SimCLR.load_from_checkpoint(weight_path, strict=False)

        elif self.model == "swav_rn50":
            check_import("pl_bolts", "swav in PretrainedSSL")
            # load resnet50 pretrained using SwAV on imagenet
            weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar"
            self.encoder = SwAV.load_from_checkpoint(weight_path, strict=False)

        elif self.model == "clip_vitb16":
            check_import("transformers", "clip_vitb16 in PretrainedSSL")
            hugging_clip = AutoModel.from_pretrained("openai/clip-vit-base-patch16")
            self.encoder = hugging_clip.get_image_features

        elif "clip" in self.model:
            check_import("clip", "clip* in PretrainedSSL")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            arch = "ViT-B/32" if "vit" in self.model else "RN50"
            model, _ = clip.load(arch, device, jit=False)
            self.encoder = model.visual  # only keep the image model

        elif "dino" in self.model:
            model = self.model.replace("rn50", "resnet50")
            self.encoder = torch.hub.load("facebookresearch/dino:main", model)

        else:
            raise ValueError(f"Unknown model={self.model}.")

        self.encoder.float()

    def reset_parameters(self):
        self.load_weights_()
