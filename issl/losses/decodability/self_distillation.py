"""Self distillation proxies to minimize R[M(X)|Z] and thus ensure decodability."""
from __future__ import annotations

import copy
import queue
from collections import Sequence
from typing import Any, Optional
import abc

import torch
import torch.distributed as dist
import torch.nn as nn
from issl.architectures import FlattenCosine, get_Architecture
from issl.helpers import RunningMean, kl_divergence, prod, queue_push_, sinkhorn_knopp, weights_init
from torch.distributions import Categorical
from torch.nn import functional as F

__all__ = [
    "PriorSelfDistillationISSL",
    "SwavSelfDistillationISSL",
    "SimSiamSelfDistillationISSL",
]

class BaseSelfDistillationISSL(nn.Module, metaclass=abc.ABCMeta):
    """Compute the ISSL loss using self distillation (i.e. approximates M(X) using current representation).

    Parameters
    ----------
    z_shape : sequence of int
        Shape of the representation.

    out_dim : float, optional
        Size of the output of the projector. Note that if  <= 1 it will be a percentage of z_dim.

    projector_kwargs : dict, optional
        Arguments to get `Projector` from `get_Architecture`.
    """

    def __init__(
            self,
            z_shape: Sequence[int],
            out_dim: float = 512,
            projector_kwargs: dict[str, Any] = {"architecture": "linear"},
    ) -> None:
        super().__init__()
        self.z_shape = [z_shape] if isinstance(z_shape, int) else z_shape
        self.z_dim = prod(self.z_shape)
        self.out_dim = out_dim if out_dim > 1 else max(10, int(self.z_dim * out_dim))

        self.projector_kwargs = self.process_shapes(projector_kwargs)
        Projector = get_Architecture(**self.projector_kwargs)
        self.projector = Projector()

        self.current_epoch = 0

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights_init(self)
        self.current_epoch = 0

    def process_shapes(self, kwargs: dict) -> dict:
        kwargs = copy.deepcopy(kwargs)  # ensure mutable object is ok
        kwargs["in_shape"] = kwargs.get("in_shape", self.z_shape)
        kwargs["out_shape"] = kwargs.get("out_shape", self.out_dim)
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

        self.current_epoch = parent.current_epoch

        if z.ndim != 2:
            raise ValueError(
                f"When using contrastive loss the representation needs to be flattened."
            )

        # shape: [batch_size, z_dim]
        z_a = parent(a, is_sample=False, is_process=True)

        loss, logs, other = self.loss(z, z_a)

        return loss, logs, other

    def loss(
        self, z: torch.Tensor, z_a: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:

        loss1, logs, other = self.asymmetric_loss(z, z_a)
        loss2, _, __ = self.asymmetric_loss(z_a, z)
        loss = (loss1 + loss2) / 2

        return loss, logs, other

    @abc.abstractmethod
    def asymmetric_loss(self, z : torch.Tensor, z_a: torch.Tensor) -> tuple[torch.Tensor, dict, dict]:
        pass

# performs SSL by having one part of the model implementing the M(X)
add_doc_prior = """              
    beta_pM_unif : float, optional
        Parameter that weights the divergence D[p_hat(M) || Unif]
        
    ema_weight_prior : bool, optional
        Weight of the exponential moving average for estimating the marginal distribution p(M). Larger means more weight 
        to the current estimate. Note that previous estimate will only be used to compute a better estimate but will not 
        be backpropagation through to avoid large memory usage for the backprop. If `None` does not use ema. 
    
    is_batchnorm_pre : bool, optional
        Whether to add a batchnorm layer before the projector / predictor.
    """

class PriorSelfDistillationISSL(BaseSelfDistillationISSL):
    __doc__ = BaseSelfDistillationISSL.__doc__ + add_doc_prior

    def __init__(
        self,
        *args,
        beta_pM_unif: float = None,
        ema_weight_prior: Optional[float] = None,
        is_batchnorm_pre: bool=False,
        **kwargs,
    ) -> None:

        super().__init__(*args,  **kwargs)
        self.beta_pM_unif = beta_pM_unif
        self.ema_weight_prior = ema_weight_prior
        self.is_batchnorm_pre = is_batchnorm_pre

        # use same arch as projector
        Predictor = get_Architecture(**self.projector_kwargs)
        self.predictor = Predictor()

        if self.is_batchnorm_pre:
            self.predictor = nn.Sequential(nn.BatchNorm1d(self.z_dim), self.predictor)
            self.projector = nn.Sequential(nn.BatchNorm1d(self.z_dim), self.projector)

        if self.ema_weight_prior is not None:
            # moving average should depend on predictor or projector => do not share
            self.ema_marginal = RunningMean(torch.ones(self.out_dim) / self.out_dim, alpha_use=self.ema_weight_prior)
            self.ema_marginal_a = RunningMean(torch.ones(self.out_dim) / self.out_dim, alpha_use=self.ema_weight_prior)
        else:
            self.ema_marginal = None
            self.ema_marginal_a = None

        self.reset_parameters()

    def asymmetric_loss(
        self, z: torch.Tensor, z_a: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:

        # shape: [batch_size, M_shape]
        M = self.predictor(z)
        M_a = self.projector(z_a)

        # shape: [batch_size, M_shape]
        loss1, logs, other = self.compare_branches(M, M_a, self.ema_marginal)
        loss2, _, __ = self.compare_branches(M_a, M, self.ema_marginal_a)
        loss = (loss1 + loss2) / 2

        return loss, logs, other

    def compare_branches(self, M, M_a, run_marginal):

        # p(M|Z). batch shape: [batch_size] ; event shape: []
        p_Mlz = Categorical(logits=M_a)

        # current p(M). batch shape: [] ; event shape: []
        hat_p_M = Categorical(probs=p_Mlz.probs.mean(0))

        mean_p_M = hat_p_M.probs.float()
        if run_marginal is not None:
            mean_p_M = run_marginal(mean_p_M)

        # p(M) moving avg. batch shape: [] ; event shape: []
        p_M = Categorical(probs=mean_p_M)

        # Unif(calM). batch shape: [] ; event shape: []
        prior = Categorical(logits=torch.ones_like(hat_p_M.probs))

        # D[\hat{p}(M) || Unif(\calM)]. shape: []
        # for unif prior same as maximizing entropy
        fit_pM_Unif = kl_divergence(p_M, prior)
        if self.ema_weight_prior is not None:
            # try to balance the decrease in gradients due to ema
            fit_pM_Unif = fit_pM_Unif / self.ema_weight_prior

        # D[p(M | Z) || q(M | Z)]. shape: [batch_size]
        # KL = - H[M|Z] - E_{p(M|Z)}[log q(M|Z)]. As you want to have a deterministic
        # p(M|Z) you want to min H[M|Z]. So min KL + H[M|Z] = - E_{p(M|Z)}[log q(M|Z)]
        fit_pMlz_qMlz = -(p_Mlz.probs * M.log_softmax(-1)).sum(-1)

        # shape: [batch_size]
        loss = fit_pMlz_qMlz + self.beta_pM_unif * fit_pM_Unif

        logs = dict(
            fit_pM_Unif=fit_pM_Unif,
            fit_pMlz_qMlz=fit_pMlz_qMlz.mean(),
            H_Mlz=p_Mlz.entropy().mean(),
            H_M=p_M.entropy(),
        )
        other = dict()

        return loss, logs, other


# do SSL by performing online clustering with hard constraint of equiprobability => learn a bank of M(X)
add_doc_swav = """
    z_shape : sequence of int
        Shape of the representation.

    n_Mx : int, optional
        Number of maximal invariant / prototypes. This is different than `out_dim` because after the clustering.
        
    freeze_Mx_epochs : int, optional
        Freeze the prototypes that many epochs from the start.
        
    temperature : float, optional
        Temperature for the softmax predictions.
        
    queue_size : int, optional
        Size of the queue to keep to enforce equipartition. Note that those examples will not be backpropagated through.
        If you do not want to use a queue then use 0.
        
    sinkhorn_kwargs : dict, optional
        Additional arguments to `sinkhorn_knopp`.
"""

# for cifar 10 check : https://github.com/facebookresearch/swav/issues/23
# and https://github.com/abhinavagarwalla/swav-cifar10
class SwavSelfDistillationISSL(BaseSelfDistillationISSL):
    __doc__ = BaseSelfDistillationISSL.__doc__ + add_doc_swav

    def __init__(
        self,
        *args,
        out_dim: int=128,
        n_Mx: int = 3000,
        freeze_Mx_epochs: int = 1,
        temperature: float = 0.1,
        queue_size: int = 15,
        sinkhorn_kwargs: dict = {},
        epoch_queue_starts : int = 15,
        **kwargs,
    ) -> None:

        self.queue_size = queue_size
        super().__init__(
            *args,
            out_dim=out_dim,
            **kwargs,
        )

        self.n_Mx = n_Mx
        self.freeze_Mx_epochs = freeze_Mx_epochs  # will be frozen in ISSL
        self.temperature = temperature
        self.epoch_queue_starts = epoch_queue_starts
        self.sinkhorn_kwargs = sinkhorn_kwargs

        self.prototypes = FlattenCosine(out_dim, self.n_Mx)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.queue = queue.Queue(maxsize=self.queue_size)

    def forward(self, *args, **kwargs):

        out = super().forward(*args,**kwargs)

        if self.is_curr_using_queue:
            queue_push_(self.queue, self._to_add_to_queue)
            del self._to_add_to_queue

        return out

    @property
    def is_curr_using_queue(self):
        return (self.queue_size > 0) and (self.current_epoch >= self.epoch_queue_starts)

    def asymmetric_loss(
        self, z: torch.Tensor, z_a: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:

        n_tgt = z_a.size(0)

        # shape: [batch_size, M_shape]
        # makes more sense to not concat because of batchnorms (but slower)
        z_proj_s = self.projector(z)

        # compute logits. shape: [batch_size, n_Mx]
        # make sure float32 and not 16
        Ms_logits = self.prototypes(z_proj_s).float()

        with torch.no_grad():
            z_proj_t = self.projector(z_a).detach()

            if self.is_curr_using_queue:
                # fill the queue with the target embedding => ensure that you still use the most recent logits.
                # + detach to avoid huge memory cost. Will only store after going through z and z_a!
                self._to_add_to_queue = z_proj_t

                if len(self.queue.queue) > 0:
                    # get logits for the queue and add them to the target ones => assignments will consider queue
                    # shape: [batch_size * queue_size, M_shape]
                    zs_proj_queue = torch.cat(list(self.queue.queue), dim=0)

                    # shape: [batch_size * queue_size + n_tgt, n_Mx]
                    z_proj_t = torch.cat([z_proj_t, zs_proj_queue], dim=0)

                else:
                    pass  # at the start has noting (and cat breaks if nothing)

            # compute logits. shape: [batch_size, n_Mx]
            Mt_logits = self.prototypes(z_proj_t).float()

            # compute assignments
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


add_doc_simsiam = """              
    predictor_kwargs : dict, optional
        Arguments to get `Predictor` from `get_Architecture`. To use no predictor set `architecture=flatten`.
    """

class SimSiamSelfDistillationISSL(BaseSelfDistillationISSL):
    __doc__ = BaseSelfDistillationISSL.__doc__ + add_doc_simsiam

    def __init__(
            self,
            *args,
            out_dim: int=2048,
            projector_kwargs: dict[str, Any] = {
                "architecture": "mlp",
                "hid_dim": 2048,
                "n_hid_layers": 2,
                "norm_layer": "batch",
                "bias": False},  # will be followed by batchnorm so drop bias
            predictor_kwargs: dict[str, Any] = {
                "architecture": "mlp",
                "hid_dim": 512,
                "n_hid_layers": 1,
                "norm_layer": "batch",
            },
            **kwargs
    ) -> None:

        super().__init__(
            *args,
            out_dim=out_dim,
            projector_kwargs=projector_kwargs,
            **kwargs,
        )

        # add batchnorm to the projector
        self.projector = nn.Sequential(self.projector, nn.BatchNorm1d(self.out_dim, affine=False))

        predictor_kwargs["in_shape"] = self.out_dim
        self.predictor_kwargs = self.process_shapes(predictor_kwargs)
        Predictor = get_Architecture(**self.predictor_kwargs)
        self.predictor = Predictor()

        self.reset_parameters()

    def loss(self, z, z_a):

        # shape: [batch_size, M_shape]
        z_proj = self.projector(z)
        z_a_proj = self.projector(z_a)

        loss, logs, other = super().loss(z_proj, z_a_proj)

        logs["std_collapse"] = F.normalize(z_proj, dim=1, p=2).std(-1).mean()
        logs["std_collapse_norm"] = logs["std_collapse"] * (z_proj.size(1) ** 0.5)  # should be one if gaussian

        return loss, logs, other

    def asymmetric_loss(
        self, z: torch.Tensor, z_a: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:

        p = self.predictor(z)
        z_a = z_a.detach()

        # shape: [batch_size]
        loss = -F.cosine_similarity(p, z_a, dim=-1)

        logs = dict()
        other = dict()

        return loss, logs, other