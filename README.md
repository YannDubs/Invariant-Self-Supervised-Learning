# Lossy Compression for Lossless Prediction [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/YannDubs/lossyless/blob/main/LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This repostiory contains pretrained weights from and the original implementation of [Improving Self-Supervised Learning by Characterizing Idealized Representations](https://github.com/YannDubs/Invariant-Self-Supervised-Learning),
which derives a simple uniying framework for invariant self-supervised learning (ISSL).
Our framework provides actionable insights into ISSL that lead to important empirical gains such as how to:
- **Simplify non-contrastive ISSL using our DISSL objective** (no momentum encoders / no stop-gradients / ... )
- **Choose the dimensionality of representations** 
- **Choose the architecture of projection probes** 
- **Choose the augmentations**

The following provides the code to reproduce our key results and load our ImageNet pretrained models.

## DISSL

(GIF)

Our DISSL objective is a very simple non-contrastive objective that outperforms previous baselines. 
Here we provide pretrained weights on ImageNet, a minimal DISSL implementation and code to reproduce our TinyImageNet results.
(Notebook to load)
(minimal implemtenat)

### Pretrained weights on ImageNet

We release our pretrained weights on torch hub. 
To load any of our model use:
```python
import torch

model = torch.hub.load('YannDubs/ISSL:main', 'resnet50')
```

Here are all available models with their respective linear probing performance on ImageNet.
They are all ResNet50 trained with a batch size of 3072.

| Epochs | Dimensionality | Multi-crop   |  ImageNet top-1 acc. |       TorchHub name |          Weights | 
|--------|----------------|--------------|---------------------:|--------------------:|-----------------:|
| 100    | 2048           | 2x224        |                 75.3 | dissl_e100_d2048_m2 |        [model]() | 
| 100    | 8192           | 2x224        |                 74.6 | dissl_e100_d8192_m2 |        [model]() | 
| 400    | 2048           | 2x224        |                 73.9 | dissl_e400_d2048_m2 |        [model]() | 
| 400    | 2048           | 2x160 + 4x96 |                 72.1 | dissl_e400_d2048_m6 |        [model]() |     
| 400    | 8192           | 2x160 + 4x96 |                 72.7 | dissl_e400_d8192_m6 |        [model]() | 
| 800    | 2048           | 2x224 + 6x96 |                 74.3 | dissl_e800_d2048_m8 |        [model]() |
| 800    | 8192           | 2x224 + 6x96 |                 70.1 | dissl_e800_d8192_m8 |        [model]() |

All the ImageNet models were ptretrained using [VISSL](www.vissl.ai) on ImageNet.
The exact commands can be seen on this (still uncleaned/undocumented) [VISSL fork](https://github.com/YannDubs/vissl) and we aim to incorporate DISSL in the main VISSL soon.

### TinyImageNet results

Once you [installed ISSL](#installation) you can reproduce the following results by running `bin/tinyimagenet/table1_distillation.sh` (right column of table 1 in the paper).
The results will be seen in `results/`




## Installation



[//]: # (## Cite)

[//]: # ()
[//]: # (You can read the full paper [here]&#40;https://arxiv.org/abs/2106.10800&#41;. Please cite our paper if you use our model:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@inproceedings{)

[//]: # (    dubois2021lossy,)

[//]: # (    title={Lossy Compression for Lossless Prediction},)

[//]: # (    author={Yann Dubois and Benjamin Bloem-Reddy and Karen Ullrich and Chris J. Maddison},)

[//]: # (    booktitle={Neural Compression: From Information Theory to Applications -- Workshop @ ICLR 2021},)

[//]: # (    year={2021},)

[//]: # (    url={https://arxiv.org/abs/2106.10800})

[//]: # (})

[//]: # (```)
