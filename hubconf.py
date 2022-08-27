
dependencies = [ "torch", "torchvision" ]

import torch
import torchvision
from hub import update_dim_resnet_ as _update_dim_resnet_

def preprocessor():
    """Preprocessor for all ISSL models."""
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _replace_dict_prefix(d, prefix, replace_with = ""):
    return { k.replace(prefix, replace_with, 1) if k.startswith(prefix) else k: v for k,v in d.items()}

def _dissl(base, dim=None, sffx="", pretrained=True, **kwargs):
    resnet = torchvision.models.__dict__[base](weights=None, **kwargs)
    if dim is not None:
        _update_dim_resnet_(resnet, z_dim=dim, bottleneck_channel=512, is_residual=True)
    resnet.fc = torch.nn.Identity()

    if pretrained:
        dir_path = "https://github.com/YannDubs/Invariant-Self-Supervised-Learning/releases/download/v1.0.0-alpha"
        ckpt_path = f"{dir_path}/dissl_{base}_d{dim}{sffx}.torch"
        state_dict = torch.hub.load_state_dict_from_url(url=ckpt_path, map_location="cpu")
        # torchvision models do not have a resizer
        state_dict = _replace_dict_prefix(state_dict, "resizer", replace_with="avgpool.0")
        resnet.load_state_dict(state_dict, strict=True)

    return resnet

def dissl_resnet50_dNone_e100_m2(pretrained=True, **kwargs):
    return _dissl(base="resnet50", dim=None, sffx="_e100_m2", pretrained=pretrained, **kwargs)

def dissl_resnet50_d8192_e100_m2(pretrained=True, **kwargs):
    return _dissl(base="resnet50", dim=8192, sffx="_e100_m2", pretrained=pretrained, **kwargs)

def dissl_resnet50_dNone_e400_m2(pretrained=True, **kwargs):
    return _dissl(base="resnet50", dim=None, sffx="_e400_m2", pretrained=pretrained, **kwargs)

def dissl_resnet50_dNone_e400_m6(pretrained=True, **kwargs):
    return _dissl(base="resnet50", dim=None, sffx="_e400_m6", pretrained=pretrained, **kwargs)

def dissl_resnet50_d8192_e400_m6(pretrained=True, **kwargs):
    return _dissl(base="resnet50", dim=8192, sffx="_e400_m6", pretrained=pretrained, **kwargs)

def dissl_resnet50_d8192_e800_m8(pretrained=True, **kwargs):
    return _dissl(base="resnet50", dim=8192, sffx="_e800_m8", pretrained=pretrained, **kwargs)