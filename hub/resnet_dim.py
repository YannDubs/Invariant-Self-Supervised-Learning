import math
import torchvision
import torch
import torch.nn as nn

__all__ = ["update_dim_resnet_"]

def update_dim_resnet_(resnet, z_dim, bottleneck_channel=None, is_residual=None):
    """Modifies inplace the representation dimensionality of an initialized torchvision resnet.

    Parameters
    ----------
    resnet : torchvision.models.ResNet
        ResNet to modify.

    z_dim : int
        New representation dimensionality.

    bottleneck_channel : int, optional
        Bottleneck to use to avoid parameter increase if increasing dimensionality.

    is_residual : bool or None, optional
        Whether to use a residual connection when using a bottleneck. Only possible
        if the current zdim is a divider of the new zdim. If `None` uses True if possible.
    """
    current_zdim = resnet.fc.in_features

    if bottleneck_channel is None:
        conv1 = nn.Conv2d(current_zdim, z_dim, kernel_size=1, bias=False)
        bn = torch.nn.BatchNorm2d(z_dim)
        weights_init(bn)
        nn.init.ones_(bn.weight)
        nn.init.zeros_(bn.bias)
        resizer = nn.Sequential(conv1, bn, torch.nn.ReLU(inplace=True))

    else:
        resizer = BottleneckExpand(current_zdim, bottleneck_channel, z_dim, is_residual=is_residual)

    resnet.avgpool = nn.Sequential(resizer, resnet.avgpool)

class BottleneckExpand(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            is_residual=None,
            norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        if is_residual is None:
            is_residual = out_channels % in_channels == 0
        if is_residual:
            assert out_channels % in_channels == 0
            self.expansion = out_channels // in_channels
        self.is_residual = is_residual

        # TODO should be sequential (not changing for backward compatibility)
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
            out += identity.repeat(1, self.expansion, 1, 1)  # add the repeated

        out = self.relu(out)

        return out

    def reset_parameters(self) -> None:
        weights_init(self)

        # using Johnson-Lindenstrauss lemma for initialization of the projection matrix
        torch.nn.init.normal_(self.conv1.weight,
                              std=1 / math.sqrt(self.conv1.weight.shape[0]))

def weights_init(module):
    """Initialize a module and all its descendents."""
    for m in module.children():
        if not init_std_modules(m):
            weights_init(m)  # go to grand children

def init_std_modules(module):
    """Initialize standard layers and return whether was initialized."""
    # all standard layers
    if isinstance(module, nn.modules.conv._ConvNd):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/d5ed58d623be27aada78035d2a19e2854f8b6437/timm/models/layers/weight_init.py
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module)
        nn.init.trunc_normal_(module, std=math.sqrt(1 / fan_in) / .87962566103423978)
        try:
            nn.init.zeros_(module.bias)
        except AttributeError:
            pass

    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        try:
            nn.init.zeros_(module.bias)
        except AttributeError:
            pass

    elif isinstance(module, nn.modules.batchnorm._NormBase):
        if module.affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    else:
        return False

    return True