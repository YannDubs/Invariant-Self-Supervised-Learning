"""Self distillation proxys to minimize R[M(X)|Z] and thus ensure decodability."""
from __future__ import annotations

import copy
import queue
from collections import Callable, Sequence
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F

from issl.architectures import get_Architecture
from issl.distributions import GumbelCategorical
from issl.helpers import kl_divergence, mean, prod, sinkhorn_knopp, weights_init

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
        Whether to process the projector. If not then this ensures the the projector and processor have
        different architectures.

    is_stop_grad : bool, optional
        Whether to stop the gradients of the part of the projector that shares parameters with the encoder.
        This will not stop gradients for self.projector.

    is_normalize_proj : bool, optional
        Whether to normalize the predictions after the predictor / projector.
    
    is_already_represented : bool, optional
        Whether the positive examples are already represented => no need to use p_ZlX again.
        In this case `p_ZlX` will be replaced by a placeholder distribution. Useful
        for clip, where the positive examples are text sentences that are already represented.

    is_proj_is_pred : bool, optional
        Whether to use the same projector as the predictor.

    predictor_kwargs : dict, optional
        Arguments to get `Predictor` from `get_Architecture`. Note that is `out_shape` is <= 1
        it will be a percentage of z_dim. To use no predictor set `mode=flatten`.

    projector_kwargs : dict, optional
        Arguments to get `Projector` from `get_Architecture`. Note that is `out_shape` is <= 1
        it will be a percentage of z_dim. To use no predictor set `mode=flatten`.
    """

    def __init__(
        self,
        z_shape: Sequence[int],
        loss: Optional[Callable],
        is_ema: bool = False,
        is_process_Mx: bool = False,
        is_stop_grad: bool = True,
        is_already_represented: bool = False,
        is_normalize_proj: bool = False,
        is_proj_is_pred: bool = False,
        predictor_kwargs: dict[str, Any] = {"mode": "linear", "out_shape": 128},
        projector_kwargs: dict[str, Any] = {"mode": "flatten", "out_shape": 128},
    ) -> None:
        super().__init__()
        self.z_shape = z_shape
        if loss is None:
            # Allows None such that module that inherit can redefine compute loss
            self.compute_loss = loss
        self.is_ema = is_ema
        self.is_process_Mx = is_process_Mx
        self.is_stop_grad = is_stop_grad
        self.is_already_represented = is_already_represented
        self.is_normalize_proj = is_normalize_proj
        self.is_proj_is_pred = is_proj_is_pred
        self.predictor_kwargs = self.process_shapes(predictor_kwargs)
        self.projector_kwargs = self.process_shapes(projector_kwargs)

        Predictor = get_Architecture(**self.predictor_kwargs)
        self.predictor = Predictor()

        if is_proj_is_pred:
            Projector = get_Architecture(**self.projector_kwargs)
            self.projector = Projector()
        else:
            self.projector = self.predictor

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights_init(self)

    def process_shapes(self, kwargs: dict) -> dict:
        kwargs = copy.deepcopy(kwargs)  # ensure mutable object is ok
        kwargs["in_shape"] = self.z_shape
        if kwargs["out_shape"] <= 1:
            kwargs["out_shape"] = max(10, int(prod(self.z_shape) * kwargs["out_shape"]))
        return kwargs

    def forward(
        self, z_hat: torch.Tensor, a: torch.Tensor, parent: Any
    ) -> tuple[torch.Tensor, dict, dict]:
        """Self distillation of examples and compute the upper bound on R[A|Z].

        Parameters
        ----------
        z_hat : Tensor shape=[batch_size, z_dim]
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

        if self.is_ema and not hasattr(parent, "ema_p_ZlX"):
            # make sure that encoder is part of the parent for EMA
            parent.ema_p_ZlX = parent.p_ZlX

        # shape: [batch_size, z_dim]
        if self.is_already_represented:
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
        hat_R_mla, logs, other = self.compute_loss(M_pred, M_a)

        return hat_R_mla, logs, other


# performs SSL by having one part of the model implementing the M(X)
add_doc = """
    beta_H_mlz : float, optional
        Whether to also minimize the entropy of p(M(x)|z), which is what you should want in theory because
        M should end up being a deterministic function. If `None` then does not compute or maximize that entropy.
        
    mode_pMlz_qMlz : {"sample_CE","sample_ST_CE","KL","CE"}, optional
        How to make q(M(X)|Z) close to p(M(X)|Z). If `"CE"` uses standard cross between 2 categoricals. If  
        `"sample_CE"` samples from p(M(X)|Z) with relaxed (gumbel) categorical and then uses standard cross entropy 
        with single sample although theoretically worst this can be useful to favor deterministic(M(X)|Z). `""sample_ST_CE"` 
        uses as before but with a straight estimator. If `"KL` uses a KL divergence. If using KL divergence this should 
        be equivalent to using adding an entropy minimizer, i.e., `beta_max_H_mlz = 1`. 
        
    divergence : {"kl_forward","kl_reverse","kl_symmetric"}, optional
        Which divergence to use between the empirical marginal p_hat(M) and the uniform categorical 
        D[p_hat(M) || Unif].         
        
    beta_pM_unif : float, optional
        Parameter that weights the divergence D[p_hat(M) || Unif]
        
    ema_weight_prior : bool, optional
        Weight of the exponential moving average. Larger means more to the current estimate. Note that non
        current estimate will only be used to compute a better estimate but will not be backpropagated through 
        to avoid large memory usage for the backprop.
    
    queue_size : bool, optional
        Size of the queue of all the p(M(X)|Z) which will be used to compute the current estimate of the marginal.
        The size is number of batches. If you do not want to use a queue then use 1. Large means memory usage ++
        in the backward pass.
    """


class PriorSelfDistillationISSL(SelfDistillationISSL):
    __doc__ = SelfDistillationISSL.__doc__ + add_doc

    def __init__(
        self,
        *args,
        beta_H_mlz: float = None,
        mode_pMlz_qMlz: float = "KL",
        divergence: str = "kl_symmetric",
        beta_pM_unif: float = 1.0,
        ema_weight_prior: Optional[float] = 0.05,
        queue_size: int = 30,
        is_normalize_proj: bool = False,
        **kwargs,
    ) -> None:
        self.queue_size = queue_size

        super().__init__(*args, is_normalize_proj=is_normalize_proj, **kwargs)
        self.beta_H_mlz = beta_H_mlz
        self.mode_pMlz_qMlz = mode_pMlz_qMlz
        self.divergence = divergence
        self.beta_pM_unif = beta_pM_unif
        self.ema_weight_prior = ema_weight_prior

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.queue = queue.Queue(maxsize=self.queue_size)

    def queue_push(self, el: torch.Tensor):
        """Pushes to the queue without going past the limit."""
        if self.queue.full():
            self.queue.get(0)
        self.queue.put_nowait(el)

    def compute_loss(
        self, M_pred: torch.Tensor, M_a: torch.Tensor
    ) -> tuple[torch.Tensor, dict, dict]:
        p_Mlz = Categorical(logits=M_a)
        curr_p_hat_M = Categorical(probs=p_Mlz.probs.mean(0))
        uniform = Categorical(logits=torch.ones_like(curr_p_hat_M.probs))

        self.queue_push(curr_p_hat_M.probs)
        mean_p_hat_m = mean(self.queue.queue)

        if self.ema_weight_prior is not None:
            if not self.hasattr(self, "mean_p_hat_m"):
                # initialize with uniform
                self.mean_p_hat_m = uniform

            alpha = self.ema_weight_prior
            assert 0.0 <= alpha <= 1.0
            mean_p_hat_m = alpha * mean_p_hat_m + (1 - alpha) * self.mean_p_hat_m
            self.mean_p_hat_m = mean_p_hat_m.detach()

        p_hat_M = Categorical(probs=mean_p_hat_m)

        # TODO: instead of KL divyou should also try with sampling from teh softmax and then use CE
        # shape: [batch_size]
        if self.mode_pMlz_qMlz == "KL":
            # KL[p(M | Z) || q(M | Z)].
            # note that KL divergence will maximize likelihood (i.e. match both) but
            # also minimize the entropy of the true distribution p(M|Z)
            fit_pMlz_qMlz = kl_divergence(p_Mlz, Categorical(logits=M_pred))
        elif self.mode_pMlz_qMlz == "CE":
            fit_pMlz_qMlz = -(p_Mlz.probs * M_pred.log_softmax(-1)).sum(-1)
        elif self.mode_pMlz_qMlz == "sample_CE":
            probs = GumbelCategorical(probs=p_Mlz.probs, is_hard=False).rsample()
            fit_pMlz_qMlz = -(probs * M_pred.log_softmax(-1)).sum(-1)
        elif self.mode_pMlz_qMlz == "sample_ST_CE":
            probs = GumbelCategorical(probs=p_Mlz.probs, is_hard=True).rsample()
            fit_pMlz_qMlz = -(probs * M_pred.log_softmax(-1)).sum(-1)
        else:
            raise ValueError(f"Unknown self.mode_pMlz_qMlz={self.mode_pMlz_qMlz}.")

        # regularizer ensures that p(M(x)) is approximately uniform. shape: []
        # note that other divergences like Wasserstein would make less sense in current categorical setting
        # because "closer" bins are not more related than "further" ones.
        if self.divergence == "kl_forward":
            fit_pM_Unif = kl_divergence(p_hat_M, uniform)
        elif self.divergence == "kl_reverse":
            fit_pM_Unif = kl_divergence(uniform, p_hat_M)
        elif self.divergence == "kl_symmetric":
            fit_pM_Unif = (
                kl_divergence(uniform, p_hat_M) + kl_divergence(p_hat_M, uniform)
            ) / 2
        else:
            raise ValueError(f"Unknown self.divergence={self.divergence}.")

        # H[M|Z]. shape: [batch_size]
        H_mlz = p_Mlz.entropy()

        # shape: [batch_size]
        loss = fit_pMlz_qMlz + self.beta_pM_unif * fit_pM_Unif

        if self.beta_H_mlz is not None:
            # Decreasing the entropy will ensure that p(M(x)|Z) is close to deterministic.
            # beta = 1 should be same as using KL divergence
            loss = loss + self.beta_H_mlz * H_mlz

        logs = dict()
        logs["fit_pM_Unif"] = fit_pM_Unif
        logs["fit_pMlz_qMlz"] = fit_pMlz_qMlz
        logs["H_mlz"] = H_mlz
        other = dict()

        return loss, logs, other


# do SSL by performing online clustering => learn a bank of M(X)
add_doc_cluster = """
    n_Mx : float, optional
        Number of clusters (i.e. \calM) to use.
        
    freeze_Mx_epochs : float, optional
        Freeze the M(X) suring that many epochs from the start.
        
    src_tgt_comparison : {"symmetric","single"}, optional
        If `"symmetric"` then compare X - A and A - X, this is standard. If `"single"` then only compares X to A, This 
        makes the most sense if A is not equivalent to X.
        
    temperature : float, optional
        Temperature for the predictions softmax predictions.
        
    queue_size : int, optional
        Size of the queue to keep to enforce equipartition. Note that those examples will not be backpropagated through.
        
    sinkhorn_kwargs : dict, optional
        Additional arguments to `sinkhorn_knopp`.
"""


class ClusterSelfDistillationISSL(PriorSelfDistillationISSL):
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
        is_proj_is_pred: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            is_normalize_proj=is_normalize_proj,
            is_proj_is_pred=is_proj_is_pred,
            **kwargs,
        )
        self.n_Mx = n_Mx
        self.freeze_Mx_epochs = freeze_Mx_epochs
        self.src_tgt_comparison = src_tgt_comparison
        self.temperature = temperature
        self.queue_size = queue_size
        self.sinkhorn_kwargs = sinkhorn_kwargs

        proj_shape = self.projector_kwargs["out_shape"]
        self.Mx_logits = nn.Linear(proj_shape, self.n_Mx, bias=False)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.queue = queue.Queue(maxsize=self.queue_size)

    def queue_push(self, el: torch.Tensor):
        """Pushes to the queue without going past the limit."""
        if self.queue.full():
            self.queue.get(0)
        self.queue.put_nowait(el)

    def forward(self, *args, **kwargs):
        # normalize M(X) weight (projected gradients descent)
        with torch.no_grad():
            w = self.Mxs.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.Mxs.weight.copy_(w)

        return super().forward(*args, **kwargs)

    def compute_loss(
        self, z_src: torch.Tensor, z_tgt: torch.Tensor
    ) -> tuple[torch.Tensor, dict, dict]:
        bs = z_src.size(0)

        # compute logits. shape: [batch_size, n_Mx]
        Ms_logits = self.Mx_logits(z_src)
        Mt_logits = self.Mx_logits(z_tgt)

        if self.src_tgt_comparison == "symmetric":
            # shape: [2*batch_size, n_Mx]
            # the src is target for target and the target is target for source => swap
            Ms_logits = torch.cat([Ms_logits, Mt_logits], dim=0)
            Mt_logits = torch.cat([Mt_logits, Ms_logits], dim=0)
        elif self.src_tgt_comparison == "single":
            pass
        else:
            raise ValueError(f"Unknown src_tgt_comparison={self.src_tgt_comparison}.")

        # use the queue.
        if self.is_queue:
            # get logits for the queue and add them to the target ones => assignments will consider queue
            z_queue = torch.cat(list(self.queue.queue), dim=0)
            Mq_logits = self.Mx_logits(z_queue)
            Mt_logits = torch.cat([Mt_logits, Mq_logits], dim=0)

            # fill the queue with the representations => ensure that you still use the most
            # recent logits. + detach to avoid huge memory cost
            self.queue_push(z_tgt.detach())

        # improve precision for the rest
        Ms_logits = Ms_logits.double()
        Mt_logits = Mt_logits.double()

        # compute assignments
        with torch.no_grad():
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1

            # shape: [(2*)batch_size, n_Mx]
            p_Mlzt = sinkhorn_knopp(
                Mt_logits, world_size=world_size, **self.sinkhorn_kwargs
            )

        # q(M(X)|Z_src). shape: [batch_size, n_Mx]
        log_q_Mlzs = (Ms_logits / self.temperature).log_softmax(-1)

        # shape: [batch_size]
        fit_pMlz_qMlz = -(p_Mlzt.probs * log_q_Mlzs).sum(-1)

        logs = dict()
        logs["fit_pMlz_qMlz"] = fit_pMlz_qMlz
        other = dict()

        return fit_pMlz_qMlz, logs, other
