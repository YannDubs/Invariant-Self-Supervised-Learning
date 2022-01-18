"""Self distillation proxies to minimize R[M(X)|Z] and thus ensure decodability."""
from __future__ import annotations

import copy
import queue
from collections import Callable, Sequence
from typing import Any, Optional
import abc

import torch
import torch.distributed as dist
import torch.nn as nn
from issl.architectures import get_Architecture
from issl.helpers import RunningMean, kl_divergence, prod, queue_push_, sinkhorn_knopp, weights_init
from torch.distributions import Categorical
from torch.nn import functional as F

__all__ = [
    "PriorSelfDistillationISSL",
    "ClusterSelfDistillationISSL",
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

    is_aux_already_represented : bool, optional
        Whether the positive examples are already represented => no need to use p_ZlX again.
        In this case `p_ZlX` will be replaced by a placeholder distribution. Useful
        for clip, where the positive examples are text sentences that are already represented.

    is_symmetric_loss : bool, optional
        Whether to use the two augmentations A,A' for both branches and symmetrize loss.

    projector_kwargs : dict, optional
        Arguments to get `Projector` from `get_Architecture`.
    """

    def __init__(
            self,
            z_shape: Sequence[int],
            out_dim: float = 1000,
            is_aux_already_represented: bool = False,
            is_symmetric_loss: bool=True,
            projector_kwargs: dict[str, Any] = {"architecture": "linear"},
    ) -> None:
        super().__init__()
        self.z_shape = [z_shape] if isinstance(z_shape, int) else z_shape
        self.out_dim = out_dim if out_dim > 1 else max(10, int(prod(self.z_shape) * out_dim))
        self.is_aux_already_represented = is_aux_already_represented
        self.is_symmetric_loss = is_symmetric_loss

        self.projector_kwargs = self.process_shapes(projector_kwargs)
        Projector = get_Architecture(**self.projector_kwargs)
        self.projector = Projector()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights_init(self)

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
        if z.ndim != 2:
            raise ValueError(
                f"When using contrastive loss the representation needs to be flattened."
            )

        # shape: [batch_size, z_dim]
        if self.is_aux_already_represented:
            # sometimes already represented, e.g., for CLIP the sentences are pre represented.
            z_a = a
        else:
            z_a = parent(a, is_sample=False, is_process=True)

        loss, logs, other = self.loss(z, z_a)

        if self.is_symmetric_loss:
            loss2, _, __ = self.loss(z_a, z)
            loss = (loss + loss2) / 2

        return loss, logs, other

    @abc.abstractmethod
    def loss(self, z : torch.Tensor, z_a: torch.Tensor) -> tuple[torch.Tensor, dict, dict]:
        pass

# performs SSL by having one part of the model implementing the M(X)
add_doc_prior = """              
    beta_pM_unif : float, optional
        Parameter that weights the divergence D[p_hat(M) || Unif]
        
    ema_weight_prior : bool, optional
        Weight of the exponential moving average for estimating the marginal distribution p(M). Larger means more weight 
        to the current estimate. Note that previous estimate will only be used to compute a better estimate but will not 
        be backpropagation through to avoid large memory usage for the backprop. If `None` does not use ema. 
        
    is_symmetric_KL_H : bool, optional
        Whether to treat both the projector and predictor with the same loss.
        
    predictor_kwargs: dict, optional
        Arguments to get `Predictor` from `get_Architecture`. 
    """

class PriorSelfDistillationISSL(BaseSelfDistillationISSL):
    __doc__ = BaseSelfDistillationISSL.__doc__ + add_doc_prior

    def __init__(
        self,
        *args,
        beta_pM_unif: float = None,
        ema_weight_prior: Optional[float] = None,
        is_symmetric_KL_H: bool=True,
        predictor_kwargs: dict[str, Any] = {"architecture": "linear"},
        **kwargs,
    ) -> None:

        super().__init__(*args,  **kwargs)
        self.beta_pM_unif = beta_pM_unif
        self.ema_weight_prior = ema_weight_prior
        self.is_symmetric_KL_H = is_symmetric_KL_H

        self.predictor_kwargs = self.process_shapes(predictor_kwargs)
        Predictor = get_Architecture(**self.predictor_kwargs)
        self.predictor = Predictor()

        if self.ema_weight_prior is not None:
            self.ema_marginal = RunningMean(torch.ones(self.out_dim) / self.out_dim, alpha=self.ema_weight_prior)

            if self.is_symmetric_KL_H:
                self.ema_marginal_a = RunningMean(torch.ones(self.out_dim) / self.out_dim, alpha=self.ema_weight_prior)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def loss(
        self, z: torch.Tensor, z_a: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:

        # shape: [batch_size, M_shape]
        M = self.predictor(z)
        M_a = self.projector(z_a)

        # shape: [batch_size, M_shape]
        loss, logs, other = self.asymmetric_loss(M, M_a)

        if self.is_symmetric_KL_H:
            loss2, _, __ = self.asymmetric_loss(M_a, M, is_a=True)
            loss = (loss + loss2) / 2

        return loss, logs, other

    def asymmetric_loss(self, M, M_a, is_a: bool=False):

        # p(M|Z). batch shape: [batch_size] ; event shape: []
        p_Mlz = Categorical(logits=M_a)

        # current p(M). batch shape: [] ; event shape: []
        hat_p_M = Categorical(probs=p_Mlz.probs.mean(0))

        mean_p_M = hat_p_M.probs.float()
        if self.ema_weight_prior is not None:
            # moving average should depend on predictor or projector => do not share
            run_marginal = self.ema_marginal_a if is_a else self.ema_marginal
            mean_p_M = run_marginal(mean_p_M)

        # p(M) moving avg. batch shape: [] ; event shape: []
        p_M = Categorical(probs=mean_p_M)

        # Unif(calM). batch shape: [] ; event shape: []
        prior = Categorical(logits=torch.ones_like(hat_p_M.probs))

        # D[\hat{p}(M) || Unif(\calM)]. shape: []
        # for unif prior same as maximizing entropy
        fit_pM_Unif = kl_divergence(p_M, prior)

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
add_doc_cluster = """
    z_shape : sequence of int
        Shape of the representation.

    n_Mx : int, optional
        Number of maximal invariant. This is different than `out_dim` because after the clustering.

    is_ema : bool, optional
        Whether to use an EMA encoder that is derived using exponential moving averaged of the predictor.

    is_stop_grad : bool, optional
        Whether to stop the gradients of the part of the projector that shares parameters with the encoder.
        This will not stop gradients for self.projector.
        
    freeze_Mx_epochs : float, optional
        Freeze the M(X) that many epochs from the start.
        
    temperature : float, optional
        Temperature for the predictions softmax predictions.
        
    queue_size : int, optional
        Size of the queue to keep to enforce equipartition. Note that those examples will not be backpropagated through.
        If you do not want to use a queue then use 0.
        
    sinkhorn_kwargs : dict, optional
        Additional arguments to `sinkhorn_knopp`.
"""

# for cifar 10 check : https://github.com/facebookresearch/swav/issues/23
# and https://github.com/abhinavagarwalla/swav-cifar10
class ClusterSelfDistillationISSL(BaseSelfDistillationISSL):
    __doc__ = BaseSelfDistillationISSL.__doc__ + add_doc_cluster

    def __init__(
        self,
        *args,
        out_dim: int=1024, # TODO check what is default value in paper
        n_Mx: int = 3000,
        is_ema: bool = True,
        is_stop_grad: bool = True,
        freeze_Mx_epochs: int = 1,
        temperature: float = 0.1,
        queue_size: int = 30,
        sinkhorn_kwargs: dict = dict(eps=0.05),
        **kwargs,
    ) -> None:

        # TODO : should add `epoch_queue_starts`
        self.queue_size = queue_size
        super().__init__(
            *args,
            out_dim=out_dim,
            **kwargs,
        )

        self.n_Mx = n_Mx
        self.is_ema = is_ema
        self.is_stop_grad = is_stop_grad
        self.freeze_Mx_epochs = freeze_Mx_epochs  # will be frozen in ISSL
        self.temperature = temperature
        self.sinkhorn_kwargs = sinkhorn_kwargs

        self.Mx_logits = nn.Linear(out_dim, self.n_Mx, bias=False)

        self.reset_parameters()


    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.queue = queue.Queue(maxsize=self.queue_size)

    def forward(self, z, a, parent):
        # normalize M(X) weight (projected gradients descent)
        with torch.no_grad():
            w = self.Mx_logits.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.Mx_logits.weight.copy_(w)

        # TODO test if EMA + callback working
        if self.is_ema and not hasattr(parent, "ema_p_ZlX"):
            # make sure that encoder is part of the parent for EMA
            parent.ema_p_ZlX = parent.p_ZlX

        return super().forward(z, a, parent)

    def loss(
        self, z: torch.Tensor, z_a: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:

        # shape: [batch_size, M_shape]
        z_proj = self.projector(z)
        z_proj_a = self.projector(z_a)

        z_src = F.normalize(z_proj, dim=1, p=2)
        z_tgt = F.normalize(z_proj_a, dim=1, p=2)

        # TODO: check if symemtric is actually what they do in the paper
        # compute logits. shape: [batch_size, n_Mx]
        tmp_Ms_logits = self.Mx_logits(z_src)
        tmp_Mt_logits = self.Mx_logits(z_tgt)

        # symmetrize. shape: [2*batch_size, n_Mx]
        Ms_logits = torch.cat([tmp_Ms_logits, tmp_Mt_logits], dim=0)
        Mt_logits = torch.cat([tmp_Mt_logits, tmp_Ms_logits], dim=0)

        if self.is_stop_grad:
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
                "is_bias": False},  # will be followed by batchnorm so drop bias
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

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def loss(self, z, z_a):
        # shape: [batch_size, M_shape]
        z = self.projector(z)
        z_a = self.projector(z_a)

        loss1 = self.asymmetric_loss(z, z_a)
        loss2 = self.asymmetric_loss(z_a, z)
        loss = (loss1 + loss2) / 2

        logs = dict()
        logs["std_collapse"] = F.normalize(z, dim=1, p=2).std(-1).mean()
        logs["std_collapse_norm"] = logs["std_collapse"] * (z.size(1) ** 0.5) # should be one if gaussian

        other = dict()

        return loss, logs, other

    def asymmetric_loss(self, z, z_a):
        p = self.predictor(z)
        z_a = z_a.detach()

        # shape: [batch_size]
        loss = F.cosine_similarity(p, z_a, dim=-1)
        return loss