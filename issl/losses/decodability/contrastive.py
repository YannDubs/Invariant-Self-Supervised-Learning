"""Contrastive proxies to minimize R[A|Z] and thus ensure decodability."""
from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Any, Optional

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

    effective_batch_size : float, optional
        Effective batch size to use for estimating InfoNCE. Larger means that more variance but less bias,
        but if too large can become not a lower bound anymore. In [1] this is (m+1)/(2*alpha), where
        +1 and / 2 comes from the fact that talking about batch size rather than sample size.
        If `None` will use the standard unweighted `effective_batch_size`. Another good possibility
        is `effective_batch_size=len_dataset` which ensures that least bias while still lower bound.

    is_normalize_proj : bool, optional
        Whether to use cosine similarity instead of dot products fot the logits of deterministic functions.
        This seems necessary for training, probably because if not norm of Z matters++ and then
        large loss in entropy bottleneck. Recommended True.

    is_aux_already_represented : bool, optional
        Whether the positive examples are already represented => no need to use p_ZlX again.
        In this case `p_ZlX` will be replaced by a placeholder distribution. Useful
        for clip, where the positive examples are text sentences that are already represented.

    src_tgt_comparison : {"all","symmetric","single"}, optional
        Which source and target pairs to compare. If `"all"` then compare both X and A against both X and A (besides
        current), this is standard. If `"symmetric"` then compare X - A and A - X (as in clip), this makes more sense
        if the representations p(Z|A) are not from the same distribution as p(Z|X). If `"single"` then only compares
        X to A, This makes the most sense if A is not equivalent to X.

    is_proj_pred_same : bool, optional
        Whether to use the projector for the predictor. Typically True in SSL, but our theory says that it should be
        False.

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
        is_train_temperature: bool = True,
        min_temperature: float = 0.01,
        is_normalize_proj: bool = True,
        effective_batch_size: Optional[float] = None,
        is_aux_already_represented: bool = False,
        is_project: bool = True,
        src_tgt_comparison: str = "all",
        is_proj_pred_same: bool = True,
        projector_kwargs: dict[str, Any] = {
            "architecture": "mlp",
            "hid_dim": 2048,
            "n_hid_layers": 2,
            "norm_layer": "batch",
            "out_shape": 128,
        },
        predictor_kwargs: dict[str, Any] = {"architecture": "linear", "out_shape": 128},
    ) -> None:
        super().__init__()
        self.z_shape = z_shape
        self.temperature = temperature
        self.is_train_temperature = is_train_temperature
        self.min_temperature = min_temperature
        self.src_tgt_comparison = src_tgt_comparison
        self.is_normalize_proj = is_normalize_proj
        self.effective_batch_size = effective_batch_size
        self.is_aux_already_represented = is_aux_already_represented
        self.is_project = is_project
        self.is_proj_pred_same = is_proj_pred_same
        self.predictor_kwargs = self.process_shapes(predictor_kwargs)
        self.projector_kwargs = self.process_shapes(projector_kwargs)

        Predictor = get_Architecture(**self.predictor_kwargs)
        self.predictor = Predictor()

        if self.is_proj_pred_same:
            self.projector = self.predictor
        else:
            Projector = get_Architecture(**self.projector_kwargs)
            self.projector = Projector()

        if self.is_train_temperature:
            self.init_temperature = temperature
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(self.init_temperature))
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights_init(self)

        if self.is_train_temperature:
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(self.init_temperature))
            )

    def process_shapes(self, kwargs: dict) -> dict:
        kwargs = copy.deepcopy(kwargs)  # ensure mutable object is ok
        kwargs["in_shape"] = self.z_shape
        if kwargs["out_shape"] <= 1:
            kwargs["out_shape"] = max(10, int(prod(self.z_shape) * kwargs["out_shape"]))
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

        # shape: [(2) * batch_size, (2) * batch_size * world_size]
        logits = self.compute_logits_p_Alz(z, a, parent)

        # shape: [(2 *) batch_size]
        hat_H_mlz, logs = self.compute_loss(logits)

        if self.src_tgt_comparison in ["all", "symmetric"]:
            # shape: [batch_size]
            hat_H_mlz = (hat_H_mlz[:batch_size] + hat_H_mlz[batch_size:]) / 2

        other = dict()

        return hat_H_mlz, logs, other

    def compute_logits_p_Alz(
        self, z: torch.Tensor, a: torch.Tensor, parent: Any
    ) -> torch.Tensor:
        """Compute the logits for the contrastive predictor p(A|Z)."""
        batch_size = z.size(0)

        # shape: [batch_size, z_dim]
        if self.is_aux_already_represented:
            # sometimes already represented, e.g., for CLIP the sentences are pre represented.
            z_a = a
        else:
            # this should actually be different for the source (sample) and the target (no sampling)
            # to be exact. i.e. should depend on `src_tgt_comparison`. But that complicated the code
            # and usually deterministic in any case
            z_a = parent(a, is_sample=False, is_process_Z=True)

        if self.src_tgt_comparison in ["all", "symmetric"]:
            # shape: [2*batch_size, z_dim]
            z_src = torch.cat([z, z_a], dim=0)
            z_tgt = torch.cat([z, z_a], dim=0)
        elif self.src_tgt_comparison == "single":
            # shape: [batch_size, z_dim]
            z_src = z
            z_tgt = z_a
        else:
            raise ValueError(f"Unknown src_tgt_comparison={self.src_tgt_comparison}.")

        # shape: [(2*)batch_size, out_shape]
        z_src = self.predictor(z_src)
        z_tgt = self.projector(z_tgt)

        if self.is_normalize_proj:
            z_src = F.normalize(z_src, dim=1, p=2)
            z_tgt = F.normalize(z_tgt, dim=1, p=2)

        # TODO test multi device
        if dist.is_available() and dist.is_initialized():
            # shape: [(2*)batch_size * world_size, out_shape]
            list_z_t = gather_from_gpus(z_tgt)
            curr_gpu = dist.get_rank()
            other_z_t = torch.cat(list_z_t[:curr_gpu] + list_z_t[curr_gpu + 1 :], dim=0)
            z_tgt = torch.cat([z_tgt, other_z_t], dim=0)

        if self.src_tgt_comparison == "all":
            # shape: [2*batch_size, 2*batch_size * world_size]
            logits = z_src @ z_tgt.T

        elif self.src_tgt_comparison == "symmetric":
            list_z_tgt = z_tgt.split(batch_size, dim=0)

            # shape: [batch_size * world_size, out_dim]
            z_tgt = torch.cat(list_z_tgt[::2], dim=0)
            z_a_tgt = torch.cat(list_z_tgt[1::2], dim=0)

            # shape: [batch_size * world_size, out_dim]
            z_src, z_a_src = z_src.chunk(2, dim=0)

            # shape: [batch_size, batch_size * world_size]
            logits = z_src @ z_tgt.T
            logits_a = z_a_src @ z_a_tgt.T

            # shape: [2*batch_size, batch_size * world_size,]
            logits = torch.cat([logits, logits_a])

        elif self.src_tgt_comparison == "single":
            # shape: [batch_size, batch_size * world_size,]
            logits = z_src @ z_tgt.T

        return logits

    def compute_loss(self, logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Computes the upper bound H_q[A|Z]."""
        n_classes = logits.size(1)  # (2*)batch_size * world_size
        new_batch_size = logits.size(0)  # (2*)batch_size
        batch_size = new_batch_size
        if self.src_tgt_comparison in ["all", "symmetric"]:
            batch_size = batch_size // 2
        device = logits.device
        arange = torch.arange(batch_size, device=device)

        if self.src_tgt_comparison == "all":
            # select all but current example.
            mask = ~torch.eye(new_batch_size, device=device).bool()
            n_to_add = n_classes - new_batch_size  # 2*batch_size * (world_size-1)
            # select all examples that are from different GPU as negatives
            ones = torch.ones(new_batch_size, n_to_add, device=device).bool()
            # shape: [2*batch_size, 2*batch_size * world_size]
            mask = torch.cat([mask, ones], dim=1)
            n_classes -= 1  # remove the current example due to masking
            logits = logits[mask].view(new_batch_size, n_classes)

            # infoNCE is essentially the same as a softmax (where weights are other z => try to predict
            # index of positive z). The label is the index of the positive z, which for each z is the
            # index that comes batch_size - 1 after it (batch_size because you concatenated) -1 because
            # you masked select all but the current z. arange takes care of idx of z which increases
            pos_idx = torch.cat([arange + batch_size - 1, arange], dim=0)

        elif self.src_tgt_comparison == "symmetric":
            # TODO test if correct
            # when both batches are only compared with the ones from other batch then you don't drop anything
            # so positives are twice on the diagonal
            pos_idx = torch.cat([arange, arange], dim=0)

        elif self.src_tgt_comparison == "single":
            # TODO test if correct
            # if you do not treat z and z_a as positives then you simply have's nothing to drop:positives are
            # the ones from the same index
            pos_idx = arange

        if self.effective_batch_size is not None:
            # TODO remove if not useful
            if self.src_tgt_comparison == "all":
                # want the reweighing so that as if the batch size was entire dataset
                # so number of negatives would be 2*dataset - 1 (one being being current)
                effective_n_classes = 2 * self.effective_batch_size - 1
            else:
                effective_n_classes = self.effective_batch_size

            # you want to multiply \sum exp(negative) in the denominator by to_mult
            # so that they have an effective weight of `effective_n_classes - 1`
            # -1 comes from the fact that only the negatives
            to_mult = (effective_n_classes - 1) / (n_classes - 1)

            # equivalent: add log(to_mult) to every negative logits
            # equivalent: add - log(to_mult) to positive logits
            to_add = -math.log(to_mult)
            to_add = to_add * torch.ones_like(logits[:, 0:1])  # correct shape
            logits.scatter_add_(1, pos_idx.unsqueeze(1), to_add)
            # when testing use `logits.gather(1, pos_idx.unsqueeze(1))` to see pos
        else:
            effective_n_classes = n_classes

        if self.is_train_temperature:
            temperature = torch.clamp(
                self.log_temperature.exp(), min=self.min_temperature
            )
        else:
            temperature = self.temperature

        # make sure 32 bits
        logits = logits.float() / temperature

        # I[Z,f(M(X))] = E[ log \frac{(N-1) exp(z^T z_p)}{\sum^{N-1} exp(z^T z')} ]
        # = log(N-1) + E[ log \frac{ exp(z^T z_p)}{\sum^{N-1} exp(z^T z')} ]
        # = log(N-1) + E[ log \frac{ exp(z^T z_p)}{\sum^{N-1} exp(z^T z')} ]
        # = log(N-1) + E[ log softmax(z^Tz_p) ]
        # = log(N-1) - E[ crossentropy(z^Tz, p) ]
        # = \hat{H}[f(M(X))] - \hat{H}[f(M(X))|Z]
        hat_H_m = math.log(effective_n_classes)
        hat_H_mlz = F.cross_entropy(logits, pos_idx, reduction="none")

        logs = dict(
            I_q_zm=(hat_H_m - hat_H_mlz.mean()), hat_H_m=hat_H_m, n_negatives=n_classes,
        )

        return hat_H_mlz, logs
