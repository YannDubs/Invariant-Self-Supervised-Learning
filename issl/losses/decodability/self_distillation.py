"""Self distillation proxies to minimize R[M(X)|Z] and thus ensure decodability."""
from __future__ import annotations

import copy
import queue
from collections import Callable, Sequence
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from issl import GumbelCategorical
from issl.architectures import get_Architecture
from issl.helpers import kl_divergence, prod, queue_push_, sinkhorn_knopp, weights_init
from torch.distributions import Categorical
from torch.nn import functional as F

__all__ = [
    "SelfDistillationISSL",
    "PriorSelfDistillationISSL",
    "ClusterSelfDistillationISSL",
]


class SelfDistillationISSL(nn.Module):
    """Compute the ISSL loss using self distillation (i.e. approximates M(X) using current representation).

    Notes
    -----
    - The key is to ensure that the M(X) you are approximating is approximately maximal (i.e. does not collapse).
    There are multiple ways of doing so: EMA, stop gradients, prior, ...

    Parameters
    ----------
    z_shape : sequence of int
        Shape of the representation.

    loss : callable, optional
        Loss with represent to which to compute R[M(X)|Z]

    is_ema : bool, optional
        Whether to use an EMA encoder that is derived using exponential moving averaged of the predictor.

    is_process_Mx : bool, optional
        Whether to process the projector. If not then this ensures the projector and processor have
        different architectures.

    is_stop_grad : bool, optional
        Whether to stop the gradients of the part of the projector that shares parameters with the encoder.
        This will not stop gradients for self.projector.

    is_normalize_proj : bool, optional
        Whether to normalize the predictions after the predictor / projector.

    is_aux_already_represented : bool, optional
        Whether the positive examples are already represented => no need to use p_ZlX again.
        In this case `p_ZlX` will be replaced by a placeholder distribution. Useful
        for clip, where the positive examples are text sentences that are already represented.

    is_pred_proj_same : bool, optional
        Whether to use the same projector as the predictor. Note that is opposite than
        in `ContrastiveISSL` to easily allow linear.

    n_Mx : float, optional
        Number of maximal invariant to predict. Note that if  <= 1 it will be a percentage of z_dim.

    predictor_kwargs : dict, optional
        Arguments to get `Predictor` from `get_Architecture`. To use no predictor set `architecture=flatten`.

    projector_kwargs : dict, optional
        Arguments to get `Projector` from `get_Architecture`. To use no predictor set `architecture=flatten`.
    """

    def __init__(
        self,
        z_shape: Sequence[int],
        loss: Optional[Callable],
        is_ema: bool = False,
        is_process_Mx: bool = False,
        is_stop_grad: bool = True,
        is_aux_already_represented: bool = False,
        is_normalize_proj: bool = False,
        is_pred_proj_same: bool = False,
        n_Mx: float = 128,
        predictor_kwargs: dict[str, Any] = {"architecture": "linear"},
        projector_kwargs: dict[str, Any] = {
            "architecture": "mlp",
            "hid_dim": 2048,
            "n_hid_layers": 2,
            "norm_layer": "batch",
        },
    ) -> None:
        super().__init__()
        self.z_shape = z_shape
        if loss is not None:
            # Allows None such that module that inherit can redefine compute loss
            self.compute_loss = loss
        self.is_ema = is_ema
        self.is_process_Mx = is_process_Mx
        self.is_stop_grad = is_stop_grad
        self.is_aux_already_represented = is_aux_already_represented
        self.is_normalize_proj = is_normalize_proj
        self.is_pred_proj_same = is_pred_proj_same
        self.n_Mx = max(10, int(prod(self.z_shape) * n_Mx))
        self.predictor_kwargs = self.process_shapes(predictor_kwargs)
        self.projector_kwargs = self.process_shapes(projector_kwargs)

        Predictor = get_Architecture(**self.predictor_kwargs)
        self.predictor = Predictor()

        if self.is_pred_proj_same:
            self.projector = self.predictor
        else:
            Projector = get_Architecture(**self.projector_kwargs)
            self.projector = Projector()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights_init(self)

    def process_shapes(self, kwargs: dict) -> dict:
        kwargs = copy.deepcopy(kwargs)  # ensure mutable object is ok
        kwargs["in_shape"] = self.z_shape
        kwargs["out_shape"] = self.n_Mx
        return kwargs

    def forward(
        self, z: torch.Tensor, a: torch.Tensor, parent: Any
    ) -> tuple[torch.Tensor, dict, dict]:
        """Self distillation of examples and compute the upper bound on R[A|Z].

        Parameters
        ----------
        z : Tensor shape=[batch_size, z_dim]
            Sampled representation.

        a : Tensor shape=[batch_size, *x_shape]
            Augmented sample.

        parent : ISSLModule, optional
            Parent module. Should have attribute `parent.p_Zlx`.

        Returns
        -------
        hat_R_MlA : torch.Tensor shape=[batch_shape]
            Estimate of R[M(X)|Z].

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """
        if z.ndim != 2:
            raise ValueError(
                f"When using contrastive loss the representation needs to be flattened."
            )

        # TODO test if EMA + callback working
        if self.is_ema and not hasattr(parent, "ema_p_ZlX"):
            # make sure that encoder is part of the parent for EMA
            parent.ema_p_ZlX = parent.p_ZlX

        # shape: [batch_size, z_dim]
        if self.is_aux_already_represented:
            # sometimes already represented, e.g., for CLIP the sentences are pre represented.
            z_a = a
        else:
            z_a = parent(a, is_sample=False, is_process_Z=self.is_process_Mx)

        if self.is_stop_grad:
            z_a = z_a.detach()

        # shape: [batch_size, M_shape]
        M_pred = self.predictor(z)
        M_a = self.projector(z_a)

        if self.is_normalize_proj:
            M_pred = F.normalize(M_pred, dim=1, p=2)
            M_a = F.normalize(M_a, dim=1, p=2)

        # shape: [batch_size, M_shape]
        hat_R_mla, logs, other = self.compute_loss(M_pred, M_a, z)

        return hat_R_mla, logs, other


# performs SSL by having one part of the model implementing the M(X)
add_doc = """                
    beta_pM_unif : float, optional
        Parameter that weights the divergence D[p_hat(M) || Unif]
        
    ema_weight_prior : bool, optional
        Weight of the exponential moving average for estimating the marginal distribution p(M). Larger means more weight 
        to the current estimate. Note that previous estimate will only be used to compute a better estimate but will not 
        be backpropagation through to avoid large memory usage for the backprop. If `None` does not use ema. 
    """


class PriorSelfDistillationISSL(SelfDistillationISSL):
    __doc__ = SelfDistillationISSL.__doc__ + add_doc

    def __init__(
        self,
        *args,
        beta_pM_unif: float = None,
        ema_weight_prior: Optional[float] = 0.5,
        is_normalize_proj: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(*args, is_normalize_proj=is_normalize_proj, **kwargs)
        self.beta_pM_unif = beta_pM_unif
        self.ema_weight_prior = ema_weight_prior

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def compute_loss(
        self, M_pred: torch.Tensor, M_a: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, dict, dict]:

        # p(M|Z). batch shape: [batch_size] ; event shape: []
        p_Mlz = Categorical(logits=M_a)

        # current p(M). batch shape: [] ; event shape: []
        hat_p_M = Categorical(probs=p_Mlz.probs.mean(0))

        # Unif(calM). batch shape: [] ; event shape: []
        uniform = Categorical(logits=torch.ones_like(hat_p_M.probs))

        mean_p_M = hat_p_M.probs.float()

        if self.ema_weight_prior is not None:
            if not hasattr(self, "ema_mean_p_M"):
                # initialize with uniform
                # TODO: should be initialized in reset_parameters as non trainable param
                self.ema_mean_p_M = uniform.probs.float()

            alpha = self.ema_weight_prior
            assert 0.0 <= alpha <= 1.0
            ema_mean_p_M = alpha * mean_p_M + (1 - alpha) * self.ema_mean_p_M
            # don't keep all the computational graph to avoid memory++
            self.ema_mean_p_M = ema_mean_p_M.detach().float()
        else:
            ema_mean_p_M = mean_p_M

        # p(M) moving avg. batch shape: [] ; event shape: []
        ema_p_M = Categorical(probs=ema_mean_p_M)

        # D[\hat{p}(M) || Unif(\calM)]. shape: []
        # this is equivalent to maximizing entropy `fit_pM_Unif = -ema_p_M.entropy()`
        # keeping the general code here in case you have a different prior on p(M)
        fit_pM_Unif = kl_divergence(ema_p_M, uniform)

        # D[p(M | Z) || q(M | Z)]. shape: [batch_size]
        # KL = - H[M|Z] - E_{p(M|Z)}[log q(M|Z)]. As you want to have a determinsitic
        # p(M|Z) you want to min H[M|Z]. So min KL + H[M|Z] = - E_{p(M|Z)}[log q(M|Z)]
        fit_pMlz_qMlz = -(p_Mlz.probs * M_pred.log_softmax(-1)).sum(-1)

        # shape: [batch_size]
        loss = fit_pMlz_qMlz + self.beta_pM_unif * fit_pM_Unif

        # H[M|Z]. shape: [batch_size]
        H_Mlz = p_Mlz.entropy()

        logs = dict(
            fit_pM_Unif=fit_pM_Unif,
            fit_pMlz_qMlz=fit_pMlz_qMlz.mean(),
            H_Mlz=H_Mlz.mean(),
        )
        other = dict()

        return loss, logs, other


# do SSL by performing online clustering => learn a bank of M(X)
add_doc_cluster = """
    n_Mx : float, optional
        Number of clusters (i.e. \calM) to use.
        
    freeze_Mx_epochs : float, optional
        Freeze the M(X) that many epochs from the start.
        
    src_tgt_comparison : {"symmetric","single"}, optional
        If `"symmetric"` then compare X - A and A - X, this is standard. If `"single"` then only compares X to A, This 
        makes the most sense if A is not equivalent to X.
        
    temperature : float, optional
        Temperature for the predictions softmax predictions.
        
    queue_size : int, optional
        Size of the queue to keep to enforce equipartition. Note that those examples will not be backpropagated through.
        If you do not want to use a queue then use 0.
        
    sinkhorn_kwargs : dict, optional
        Additional arguments to `sinkhorn_knopp`.
"""


class ClusterSelfDistillationISSL(SelfDistillationISSL):
    __doc__ = SelfDistillationISSL.__doc__ + add_doc_cluster

    def __init__(
        self,
        *args,
        n_Mx: int = 3000,
        freeze_Mx_epochs: int = 1,
        src_tgt_comparison: str = "symmetric",
        temperature: float = 0.1,
        queue_size: int = 30,
        sinkhorn_kwargs: dict = dict(eps=0.05),
        is_normalize_proj: bool = True,
        # clustering is non differentiable so no grad besides if `src_tgt_comparison=="symmetric"`
        is_pred_proj_same: bool = True,
        is_stop_grad: bool = True,
        **kwargs,
    ) -> None:

        # TODO : should add `epoch_queue_starts`
        self.queue_size = queue_size
        super().__init__(
            *args,
            is_normalize_proj=is_normalize_proj,
            is_pred_proj_same=is_pred_proj_same,
            is_stop_grad=False,  # don't stop gradient in target if using symmetric
            **kwargs,
        )
        self.n_Mx = n_Mx
        self.freeze_Mx_epochs = freeze_Mx_epochs  # will be frozen in ISSL
        self.src_tgt_comparison = src_tgt_comparison
        self.temperature = temperature
        self.sinkhorn_kwargs = sinkhorn_kwargs
        self.is_actual_stop_grad = is_stop_grad

        proj_shape = self.projector_kwargs["out_shape"]
        self.Mx_logits = nn.Linear(proj_shape, self.n_Mx, bias=False)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.queue = queue.Queue(maxsize=self.queue_size)

    def forward(self, *args, **kwargs):
        # normalize M(X) weight (projected gradients descent)
        with torch.no_grad():
            w = self.Mx_logits.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.Mx_logits.weight.copy_(w)

        return super().forward(*args, **kwargs)

    def compute_loss(
        self, z_src: torch.Tensor, z_tgt: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, dict, dict]:

        # compute logits. shape: [batch_size, n_Mx]
        tmp_Ms_logits = self.Mx_logits(z_src)
        tmp_Mt_logits = self.Mx_logits(z_tgt)

        if self.src_tgt_comparison == "symmetric":
            # shape: [2*batch_size, n_Mx]
            # the src is target for target and the target is target for source => swap
            Ms_logits = torch.cat([tmp_Ms_logits, tmp_Mt_logits], dim=0)
            Mt_logits = torch.cat([tmp_Mt_logits, tmp_Ms_logits], dim=0)
            n_tgt = Mt_logits.size(0)
        elif self.src_tgt_comparison == "single":
            Ms_logits = tmp_Ms_logits
            Mt_logits = tmp_Mt_logits
            n_tgt = Mt_logits.size(0)
        else:
            raise ValueError(f"Unknown src_tgt_comparison={self.src_tgt_comparison}.")

        if self.is_actual_stop_grad:
            Mt_logits = Mt_logits.detach()

        # use the queue.
        if self.queue_size > 0:
            if len(self.queue.queue) == 0:
                # for first step ensure has at least one element
                queue_push_(self.queue, tmp_Mt_logits.detach())

            # get logits for the queue and add them to the target ones => assignments will consider queue
            # shape: [batch_size * queue_size, n_Mx]
            Mq_logits = torch.cat(list(self.queue.queue), dim=0)
            # shape: [batch_size * queue_size + n_tgt, n_Mx]
            Mt_logits = torch.cat([Mt_logits, Mq_logits], dim=0)

            # fill the queue with the representations => ensure that you still use the most
            # recent logits. + detach to avoid huge memory cost
            queue_push_(self.queue, tmp_Mt_logits.detach())

        # make sure float32 and not 16
        Ms_logits = Ms_logits.float()
        Mt_logits = Mt_logits.float()

        # compute assignments
        with torch.no_grad():
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1

            # shape: [(batch_size * queue_size) + n_samples, n_Mx]
            p_Mlzt = sinkhorn_knopp(
                Mt_logits, world_size=world_size, **self.sinkhorn_kwargs
            )

            # chose only the target ones. shape: [ n_tgt, n_Mx]
            p_Mlzt = p_Mlzt[:n_tgt]

        # log q(M(X)|Z_src). shape: [n_tgt, n_Mx]
        log_q_Mlzs = (Ms_logits / self.temperature).log_softmax(-1)

        # shape: [n_tgt]
        fit_pMlz_qMlz = -(p_Mlzt * log_q_Mlzs).sum(-1)

        logs = dict(fit_pMlz_qMlz=fit_pMlz_qMlz.mean())
        other = dict()

        return fit_pMlz_qMlz, logs, other
