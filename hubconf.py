
dependencies = [ "torch", "torchvision" ]

import torch
import torchvision
from hub import update_dim_resnet_ as _update_dim_resnet_
from torchvision.models import _api, _utils


def preprocessor():
    """Preprocessor for all ISSL models."""
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class DISSL_ResNet50DNone_Weights(_api.WeightsEnum):
    DISSL_RESNET50_DNONE_E100_M2 = _api.Weights(
        url="https://github.com/YannDubs/Invariant-Self-Supervised-Learning/releases/download/v1.0.0-alpha"
            "/dissl_resnet50_dNone_e100_m2.torch",
        transforms=preprocessor,
        meta={
            "num_params": 23508032,
            "recipe": "https://github.com/YannDubs/vissl",  #TODO update once on main VISSL
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 66.4,
                }
            },
            "_docs": "ResNet50 with standard (2048) dimensionality trained on ImageNet-1K for 100 epochs with DISSL loss "
                     "and 2x224 crops (ie no multi-crops).",
        },
    )

    DISSL_RESNET50_DNONE_E400_M2 = _api.Weights(
        url="https://github.com/YannDubs/Invariant-Self-Supervised-Learning/releases/download/v1.0.0-alpha/"
            "dissl_resnet50_dNone_e400_m2.torch",
        transforms=preprocessor,
        meta={
            "num_params": 23508032,
            "recipe": "https://github.com/YannDubs/vissl",  # TODO update once on main VISSL
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 70.4,
                }
            },
            "_docs": "ResNet50 with standard (2048) dimensionality trained on ImageNet-1K for 400 epochs with DISSL loss "
                     "and 2x224 crops (ie no multi-crops).",
        },
    )

    DISSL_RESNET50_DNONE_E400_M6 = _api.Weights(
        url="https://github.com/YannDubs/Invariant-Self-Supervised-Learning/releases/download/v1.0.0-alpha"
            "/dissl_resnet50_dNone_e400_m6.torch",
        transforms=preprocessor,
        meta={
            "num_params": 23508032,
            "recipe": "https://github.com/YannDubs/vissl",  # TODO update once on main VISSL
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.5,
                }
            },
            "_docs": "ResNet50 with standard (2048) dimensionality trained on ImageNet-1K for 400 epochs with DISSL loss "
                     "and 2x160+4*96 multi-crops.",
        },
    )

    DEFAULT = DISSL_RESNET50_DNONE_E400_M6

class DISSL_ResNet50D8192_Weights(_api.WeightsEnum):
    DISSL_RESNET50_D8192_E100_M2 = _api.Weights(
        url="https://github.com/YannDubs/Invariant-Self-Supervised-Learning/releases/download/v1.0.0-alpha"
            "/dissl_resnet50_d8192_e100_m2.torch",
        transforms=preprocessor,
        meta={
            "num_params": 31128640,
            "recipe": "https://github.com/YannDubs/vissl",  #TODO update once on main VISSL
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 67.6,
                }
            },
            "_docs": "ResNet50 with increased (8192) dimensionality trained on ImageNet-1K for 100 epochs with DISSL loss "
                     "and 2x224 crops (ie no multi-crops).",
        },
    )

    DISSL_RESNET50_D8192_E400_M6 = _api.Weights(
        url="https://github.com/YannDubs/Invariant-Self-Supervised-Learning/releases/download/v1.0.0-alpha"
            "/dissl_resnet50_d8192_e400_m6.torch",
        transforms=preprocessor,
        meta={
            "num_params": 31128640,
            "recipe": "https://github.com/YannDubs/vissl",  # TODO update once on main VISSL
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 72.6,
                }
            },
            "_docs": "ResNet50 with increased (8192) dimensionality trained on ImageNet-1K for 400 epochs with DISSL loss "
                     "and 2x160+4*96 multi-crops.",
        },
    )

    DISSL_RESNET50_D8192_E800_M8 = _api.Weights(
        url="https://github.com/YannDubs/Invariant-Self-Supervised-Learning/releases/download/v1.0.0-alpha"
            "/dissl_resnet50_d8192_e800_m8.torch",
        transforms=preprocessor,
        meta={
            "num_params": 31128640,
            "recipe": "https://github.com/YannDubs/vissl",  # TODO update once on main VISSL
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 72.9,
                }
            },
            "_docs": "ResNet50 with increased (8192) dimensionality trained on ImageNet-1K for 800 epochs with DISSL "
                     "loss and 2x224+6*96 multi-crops.",
        },
    )

    DEFAULT = DISSL_RESNET50_D8192_E800_M8


def _replace_dict_prefix(d, prefix, replace_with = ""):
    return { k.replace(prefix, replace_with, 1) if k.startswith(prefix) else k: v for k,v in d.items()}

def _dissl(base, dim=None, weights=None, progress=True, **kwargs):
    resnet = torchvision.models.__dict__[base](**kwargs)
    if dim is not None:
        _update_dim_resnet_(resnet, z_dim=dim, bottleneck_channel=512, is_residual=True)
    resnet.fc = torch.nn.Identity()

    if weights is not None:
        state_dict = weights.get_state_dict(progress=progress)
        # torchvision models do not have a resizer
        state_dict = _replace_dict_prefix(state_dict, "resizer", replace_with="avgpool.0")
        resnet.load_state_dict(state_dict, strict=True)

    return resnet

@_utils.handle_legacy_interface(weights=("pretrained", DISSL_ResNet50DNone_Weights.DEFAULT))
def dissl_resnet50_dNone(*, weights=None, **kwargs):
    weights = DISSL_ResNet50DNone_Weights.verify(weights)
    return _dissl(base="resnet50", weights=weights, **kwargs)

@_utils.handle_legacy_interface(weights=("pretrained", DISSL_ResNet50D8192_Weights.DEFAULT))
def dissl_resnet50_d8192(*, weights=None, **kwargs):
    weights = DISSL_ResNet50D8192_Weights.verify(weights)
    return _dissl(base="resnet50", dim=8192, weights=weights, **kwargs)