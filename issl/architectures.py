import logging
from functools import partial

import torch

try:
    import clip
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

logger = logging.getLogger(__name__)


### ClASSES ###
