# Lossy Compression for Lossless Prediction [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/YannDubs/lossyless/blob/main/LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

[Loading]

This repostiory contains pretrained weights from and the original implementation of [Improving Self-Supervised Learning by Characterizing Idealized Representations](https://github.com/YannDubs/Invariant-Self-Supervised-Learning),
which derives a simple uniying framework for invariant self-supervised learning (ISSL).
Our framework provides actionable insights into ISSL that lead to important empirical gains such as how to:
- [**Simplify non-contrastive ISSL using our DISSL objective**](#dissl-tinyimagenet) (no momentum encoders / no stop-gradients / ... )
- [**Choose the dimensionality of representations**](#dimensionality) 
- [**Choose the architecture of projection probes**](#projection-heads) 
- [**Choose the augmentations**](#augmentations)

The following provides the code to reproduce our key results and load our ImageNet pretrained models.
  
## DISSL

(GIF)

Our DISSL objective is a very simple non-contrastive objective that outperforms previous baselines. 
We provide pretrained weights on ImageNet and a minimal DISSL implementation (see [![Minimal training of DISSL](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YannDubs/lossyless/blob/main/notebooks/minimal_code.ipynb)).

We release our pretrained weights on torch hub. 
To load any of our model use:
```python
import torch

model = torch.hub.load('YannDubs/ISSL:main', 'resnet50')
```

Here are all available models with their respective linear probing performance on ImageNet.
They are all ResNet50 trained with a batch size of 3072.

| Epochs | Dimensionality | Multi-crop   | ImageNet top-1 acc. |       TorchHub name |          Weights | 
|--------|----------------|--------------|--------------------:|--------------------:|-----------------:|
| 100    | 2048           | 2x224        |                66.3 | dissl_e100_d2048_m2 |        [model]() | 
| 100    | 8192           | 2x224        |                67.7 | dissl_e100_d8192_m2 |        [model]() | 
| 400    | 2048           | 2x224        |                70.4 | dissl_e400_d2048_m2 |        [model]() | 
| 400    | 2048           | 2x160 + 4x96 |                71.4 | dissl_e400_d2048_m6 |        [model]() |     
| 400    | 8192           | 2x160 + 4x96 |                72.6 | dissl_e400_d8192_m6 |        [model]() | 
| 800    | 2048           | 2x224 + 6x96 |                     | dissl_e800_d2048_m8 |        [model]() |
| 800    | 8192           | 2x224 + 6x96 |                     | dissl_e800_d8192_m8 |        [model]() |


## Reproducing main results 

To reproduce our key TinyImageNet results you need to `pip install -r requirements.txt` and run the desired script in `bin/tinyimagenet/*.sh`.
Details below.

For our ImageNet models we used [VISSL](www.vissl.ai). The exact commands can be seen on this (still uncleaned/undocumented) [VISSL fork](https://github.com/YannDubs/vissl) and we aim to incorporate DISSL in the main VISSL soon.


### DISSL TinyImageNet

Once you [installed ISSL](#installation) you can reproduce the following results by running `bin/tinyimagenet/table1_distillation.sh` (subset of right column of table 1 in the paper).
The results will be seen in `results/exp_table1_distillation`

| Model    | TinyImageNet Linear probing acc. |
|:---------|---------------------------------:|
| DINO     |                            43.6% |
| DISSL    |                            45.9% |
| + dim.   |                            47.7% |
| + epochs |                            49.7% |
 | + aug.   |                            50.1% |

WandB monitoring curves: [see here](https://wandb.ai/issl/issl_opensource/groups/table1_distillation)

### Dimensionality

In our paper we characterize exactly the minimal and sufficient dimensionality depending on the probing architecture.
For linear probes it's much larger than standard dimensionalities, which suggests that one would gain important gains by increasing dimensionality. 
Figure 7c of our paper shows empirically that this is indeed the case.
To reproduce a similar figure (single seed) run `bin/tinyimagenet/fig7c_dimensions.sh`.
The following figure will then be saved in `results/exp_fig7c_dimensions`.

(Figure)

### Projection heads


In our paper, we prove that one of the two projection heads needs to have the same architecture as the dowsntream probe.
This is to ensure that the SSL representations are pretrained the same way as they will be used in downstream tasks.

(GIF)

This is the difference between our CISSL and SimCLR. 
The left column in Table 1 of our paper shows empirically that this improves performance.
To reproduce a similar table (single seed) run `bin/tinyimagenet/table1_contrastive.sh`.
The following results will be seen in `results/exp_table1_distillation`

| Model    | TinyImageNet Linear probing acc. |
|:---------|---------------------------------:|
| SimCLR   |                            44.6% |
| CISSL    |                            45.8% |
| + dim.   |                            47.5% |
| + epochs |                            48.6% |
 | + aug.   |                            50.1% |



### Augmentations

In our paper we characterize exactly optimal sample efficiency as a function of how coarse the equivalence class induced by augmentations are.
In particular, our theory suggests that stronger label-preserving augmentations improve performance.
Figure 7a of our paper shows empirically that this is indeed the case.
To reproduce a similar figure (single seed) run `bin/tinyimagenet/fig7a_augmentations.sh`.
The following figure will then be saved in `results/exp_fig7a_augmentations`.

(Figure)



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
