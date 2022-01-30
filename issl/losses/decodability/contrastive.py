"""Contrastive proxies to minimize R[A|Z] and thus ensure decodability."""
from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

from issl.architectures import get_Architecture
from issl.helpers import gather_from_gpus, prod, weights_init

__all__ = ["ContrastiveISSL"]


# TODO should probably write it as a subclass of self distillation because you are reusing a lot of the
class ContrastiveISSL(nn.Module):
    """Computes the ISSL loss using contrastive variational bound (i.e. with positive and negative examples).

    Notes
    -----
    - parts of code taken from https://github.com/lucidrains/contrastive-learner

    Parameters
    ----------
    z_shape : sequence of int
        Shape of the representation.

    temperature : float, optional
        Temperature scaling in InfoNCE. Recommended less than 1.

    is_train_temperature : bool, optional
        Whether to treat the temperature as a parameter. Uses the same same as CLIP.
        If so `temperature` will be used as the initialization.

    min_temperature : float, optional
        Lower bound on the temperature. Only if `is_train_temperature`.

    is_pred_proj_same : bool, optional
        Whether to use the projector for the predictor. Typically, True in SSL, but our theory says that it should be
        False.

    is_self_contrastive : bool, optional
        Whether to keep current example in batch when performing contrastive learning and then have to predict
        proba 0.5 to current and positive.

    is_batchnorm_post : bool, optional
        Whether to add a batchnorm layer after the predictor / projector.

    is_batchnorm_pre : bool, optional
        Whether to add a batchnorm layer pre the predictor / projector.

    projector_kwargs : dict, optional
        Arguments to get `Projector` from `get_Architecture`. Note that is `out_shape` is <= 1
        it will be a percentage of z_dim. To use no projector set `mode=Flatten`.

    predictor_kwargs : dict, optional
        Arguments to get `Predictor` from `get_Architecture`. Note that is `out_shape` is <= 1
        it will be a percentage of z_dim. To use no predictor set `mode=Flatten`.

    References
    ----------
    [1] Song, Jiaming, and Stefano Ermon. "Multi-label contrastive predictive coding." Advances in
    Neural Information Processing Systems 33 (2020).
    """

    def __init__(
        self,
        z_shape: Sequence[int],
        temperature: float = 0.07,
        is_train_temperature: bool = False,
        min_temperature: float = 0.01,
        is_pred_proj_same: bool = False,
        is_self_contrastive: bool = False,
        is_batchnorm_pre: bool = False,
        is_batchnorm_post: bool = False,
        projector_kwargs: dict[str, Any] = {
            "architecture": "mlp",
            "hid_dim": 2048,
            "n_hid_layers": 2,
            "norm_layer": "batch",
        },
        predictor_kwargs: dict[str, Any] = {"architecture": "flatten"},
    ) -> None:
        super().__init__()
        self.z_shape = [z_shape] if isinstance(z_shape, int) else z_shape
        self.z_dim = prod(self.z_shape)
        self.is_train_temperature = is_train_temperature
        self.min_temperature = min_temperature
        self.is_pred_proj_same = is_pred_proj_same
        self.is_batchnorm_pre = is_batchnorm_pre
        self.is_batchnorm_post = is_batchnorm_post
        self.predictor_kwargs = self.process_kwargs(predictor_kwargs)
        self.projector_kwargs = self.process_kwargs(projector_kwargs)
        self.is_self_contrastive = is_self_contrastive

        if self.is_pred_proj_same:
            self.is_self_contrastive = False

        Projector = get_Architecture(**self.projector_kwargs)
        self.projector = self.add_batchnorms(Projector())

        if self.is_pred_proj_same:
            self.predictor = self.projector
        else:
            Predictor = get_Architecture(**self.predictor_kwargs)
            self.predictor = self.add_batchnorms(Predictor())

        if self.is_train_temperature:
            self.init_temperature = temperature
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(self.init_temperature))
            )
        else:
            self._temperature = temperature

        self.reset_parameters()

    def add_batchnorms(self, head):
        if self.is_batchnorm_pre:
            head = nn.Sequential(nn.BatchNorm1d(self.z_dim), head)
        if self.is_batchnorm_post:
            head = nn.Sequential(head, nn.BatchNorm1d(self.out_dim))
        return head

    def reset_parameters(self) -> None:
        weights_init(self)

        if self.is_train_temperature:
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(self.init_temperature))
            )

    def process_kwargs(self, kwargs: dict) -> dict:
        kwargs = copy.deepcopy(kwargs)  # ensure mutable object is ok
        kwargs["in_shape"] = self.z_shape

        if "out_shape" not in kwargs:
            kwargs["out_shape"] = prod(self.z_shape)
        elif kwargs["out_shape"] <= 1:
            kwargs["out_shape"] = max(10, int(self.z_dim * kwargs["out_shape"]))

        self.out_dim = kwargs["out_shape"]

        if self.is_batchnorm_post and kwargs["architecture"] in ["mlp", "linear"]:
            kwargs["bias"] = False  # no bias when batchorm

        return kwargs

    def forward(
        self, z: torch.Tensor, a: torch.Tensor, parent: Any
    ) -> tuple[torch.Tensor, dict, dict]:
        """Contrast examples and compute the upper bound on R[A|Z].
    
        Parameters
        ----------
        z : Tensor shape=[batch_size, z_dim]
            Reconstructed representations.

        a : Tensor shape=[batch_size, *x_shape]
            Augmented sample.

        parent : ISSLModule, optional
            Parent module. Should have attribute `parent.p_ZlX`.
    
        Returns
        -------
        loss : torch.Tensor shape=[batch_shape]
            Estimate of the loss.
    
        logs : dict
            Additional values to log.
    
        other : dict
            Additional values to return.
        """

        if z.ndim != 2:
            raise ValueError(
                f"When using contrastive loss the representation needs to be flattened."
            )

        batch_size, z_dim = z.shape

        # shape: [batch_size, z_dim]
        z_a = parent(a, is_sample=False, is_process=True)

        # shape: [(2) * batch_size, (2) * batch_size * world_size]
        logits = self.compute_logits_p_Alz(z, z_a)

        # shape: [(2 *) batch_size]
        hat_H_mlz, logs = self.compute_loss(logits)

        # shape: [batch_size]
        hat_H_mlz = (hat_H_mlz[:batch_size] + hat_H_mlz[batch_size:]) / 2

        other = dict()

        return hat_H_mlz, logs, other

    def compute_logits_p_Alz(
        self, z: torch.Tensor, z_a: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the logits for the contrastive predictor p(A|Z)."""

        # shape: [2*batch_size, z_dim]
        z_src = torch.cat([z, z_a], dim=0)
        z_tgt = torch.cat([z, z_a], dim=0)

        # shape: [(2*)batch_size, out_shape]
        z_src = self.predictor(z_src)
        z_tgt = self.projector(z_tgt)

        # will use cosine similarity
        z_src = F.normalize(z_src, dim=1, p=2)
        z_tgt = F.normalize(z_tgt, dim=1, p=2)

        # TODO test multi device
        if dist.is_available() and dist.is_initialized():
            # shape: [(2*)batch_size * world_size, out_shape]
            list_z_t = gather_from_gpus(z_tgt)
            curr_gpu = dist.get_rank()
            other_z_t = torch.cat(list_z_t[:curr_gpu] + list_z_t[curr_gpu + 1 :], dim=0)
            z_tgt = torch.cat([z_tgt, other_z_t], dim=0)

        # shape: [(2*)batch_size, (2*)batch_size * world_size]
        logits = z_src @ z_tgt.T

        return logits

    @property
    def temperature(self):
        if self.is_train_temperature:
            temperature = torch.clamp(
                self.log_temperature.exp(), min=self.min_temperature
            )
        else:
            temperature = self._temperature
        return temperature

    def compute_loss(self, logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Computes the upper bound H_q[A|Z]."""
        n_classes = logits.size(1)  # (2*)batch_size * world_size
        new_batch_size = logits.size(0)  # (2*)batch_size
        batch_size = new_batch_size // 2
        device = logits.device
        arange = torch.arange(batch_size, device=device)

        # make sure 32 bits
        logits = logits.float() / self.temperature

        if not self.is_self_contrastive:
            # select all but current example.
            mask = ~torch.eye(new_batch_size, device=device).bool()
            n_to_add = n_classes - new_batch_size  # 2*batch_size * (world_size-1)
            # select all examples that are from different GPU as negatives
            ones = torch.ones(new_batch_size, n_to_add, device=device).bool()
            # shape: [2*batch_size, 2*batch_size * world_size]
            mask = torch.cat([mask, ones], dim=1)
            n_classes -= 1  # remove the current example due to masking

            # infoNCE is essentially the same as a softmax (where weights are other z => try to predict
            # index of positive z). The label is the index of the positive z, which for each z is the
            # index that comes batch_size - 1 after it (batch_size because you concatenated) -1 because
            # you masked select all but the current z. arange takes care of idx of z which increases
            pos_idx = torch.cat([arange + batch_size - 1, arange], dim=0)

            # I[Z,f(M(X))] = E[ log \frac{(N-1) exp(z^T z_p)}{\sum^{N-1} exp(z^T z')} ]
            # = log(N-1) + E[ log \frac{ exp(z^T z_p)}{\sum^{N-1} exp(z^T z')} ]
            # = log(N-1) + E[ log \frac{ exp(z^T z_p)}{\sum^{N-1} exp(z^T z')} ]
            # = log(N-1) + E[ log softmax(z^Tz_p) ]
            # = log(N-1) - E[ crossentropy(z^Tz, p) ]
            # = \hat{H}[f(M(X))] - \hat{H}[f(M(X))|Z]
            hat_H_mlz = F.cross_entropy(logits[mask].view(new_batch_size, n_classes),
                                        pos_idx,
                                        reduction="none")

        else:
            # here just predict 0.5 probability for both => no masking needed. This should somehow ensure invariance
            # all the add examples have 0 probability under p
            log_q = logits.log_softmax(-1)[:, :new_batch_size]
            # only keep the probability you assign to the 2 positives then sum and divide by 2
            # essentially multiply and sum by p giving mass 0.5 to each
            mask = torch.eye(batch_size, device=device).bool().repeat(2, 2)
            log_q = log_q[mask].view(new_batch_size, 2)
            hat_H_mlz = - log_q.sum(1) / 2

        hat_H_m = math.log(n_classes)
        logs = dict(
            I_q_zm=(hat_H_m - hat_H_mlz.mean()),
            hat_H_m=hat_H_m,
            n_negatives=n_classes,
        )

        return hat_H_mlz, logs
