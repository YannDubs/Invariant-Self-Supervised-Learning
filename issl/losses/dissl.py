"""Distillation proxy to ISSL log loss."""
from __future__ import annotations

import copy
from typing import Any, Optional

import torch
import torch.nn as nn
from issl.architectures import get_Architecture
from issl.helpers import weights_init, mean
from torch.distributions import Categorical
from torch.nn import functional as F

class DISSL(nn.Module):
    """Compute the ISSL loss using self distillation (i.e. approximates M(X) using current representation).

    Parameters
    ----------
    z_dim : int
        Dimensionality of the representation.

    n_equivalence_classes : int, optional
        Number of equivalence classes (C in the paper)

    lambda_maximality : float, optional
        Parameter that weights the divergence D[p_hat(M) || Unif]

    projector_kwargs : dict, optional
        Arguments to get `Projector` from `get_Architecture`.
    """

    def __init__(
        self,
        z_dim: int,
        n_equivalence_classes: float = 16384,
        lambda_maximality: float = 2.3,
        beta_det_inv: float=0.8,
        temperature: float=1.0,
        temperature_assign:  Optional[float] = None,
        projector_kwargs: dict[str, Any] = {
            "architecture": "mlp",
            "bottleneck_size": 512,
            "hid_dim": 1024,
            "n_hid_layers": 1,
            "is_cosine": True,
            "norm_layer": "batch"
        },
        predictor_kwargs: dict[str, Any] = {
                               "architecture": "linear",
                           },
    ) -> None:

        super().__init__()
        self.z_dim = z_dim
        self.n_equivalence_classes = n_equivalence_classes
        self.lambda_maximality = lambda_maximality
        self.beta_det_inv = beta_det_inv
        self.temperature = temperature
        self.temperature_assign = temperature_assign or self.temperature / 2

        # code ready for multi crops
        self.crops_assign = 2
        self.crops_pred = 2

        self.projector_kwargs = self.process_shapes(projector_kwargs)
        Projector = get_Architecture(**self.projector_kwargs)
        self.projector = Projector()

        self.predictor_kwargs = self.process_shapes(predictor_kwargs)
        Predictor = get_Architecture(**self.predictor_kwargs)
        self.predictor = Predictor()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights_init(self)

    def process_shapes(self, kwargs: dict) -> dict:
        kwargs = copy.deepcopy(kwargs)  # ensure mutable object is ok
        kwargs["in_shape"] = kwargs.get("in_shape", self.z_dim)
        kwargs["out_shape"] = kwargs.get("out_shape", self.n_equivalence_classes)
        return kwargs

    def get_Mx(self, z_tgt):
        """Return the parameters of the categorical dist of predicted M(X). If entropy is 0 then this is Mx."""
        logits_assign = self.projector(z_tgt).float() / self.temperature_assign
        Mx = F.softmax(logits_assign, dim=-1)
        return Mx

    def forward(self, z: torch.Tensor, z_tgt: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Self distillation of examples and compute the upper bound on R[A|Z].

        Parameters
        ----------
        z : Tensor shape=[2 * batch_size, z_dim]
            Sampled representation.

        z_tgt : Tensor shape=[2 * batch_size, z_dim]
            Representation from the other branch.

        Returns
        -------
        loss : torch.Tensor shape=[]

        logs : dict
            Additional values to monitor.
        """

        # shape: [batch_size, M_shape]. Make sure not half prec
        logits_assign = self.projector(z_tgt).float() / self.temperature_assign
        logits_predict = self.predictor(z).float() / self.temperature

        all_p_Mlz = F.softmax(logits_assign, dim=-1).chunk(self.crops_assign)
        all_log_p_Mlz = F.log_softmax(logits_assign, dim=-1).chunk(self.crops_assign)
        all_log_q_Mlz = F.log_softmax(logits_predict, dim=-1).chunk(self.crops_pred)

        list_CE_pMlz_qMlza = []
        list_CE_pMlz_pMlza = []
        list_CE_pMlz_qMlz = []
        list_H_M = []
        # for loop enables multi crop augmentations if needed
        for i_p, p_Mlz in enumerate(all_p_Mlz):

            ##### MAXIMALITY #####
            # current marginal estimate p(M). batch shape: [] ; event shape: []
            p_M = p_Mlz.mean(0)

            # H[\hat{M}]. shape: []
            # for unif prior same as maximizing entropy.
            list_H_M += [Categorical(probs=p_M).entropy()]
            #############################

            ##### INVARIANCE and DETERMINISM #####
            for i_log_p, log_p_Mlza in enumerate(all_log_p_Mlz):
                if i_p == i_log_p:
                    continue  # skip if same view

                # E_{t(M|Z)}[- log t(M|\tilde{Z})]. shape: []
                list_CE_pMlz_pMlza += [- (p_Mlz * log_p_Mlza).sum(-1).mean()]
            #######################################

            ##### DISTILLATION #####
            for i_q, log_q_Mlza in enumerate(all_log_q_Mlz):
                if i_p == i_q:
                    # only used for monitoring
                    list_CE_pMlz_qMlz += [- (p_Mlz.detach() * log_q_Mlza.detach()).sum(-1).mean()]
                    continue   # skip if same view

                # E_{t(M|Z)}[- log s(M|\tilde{Z})]. shape: []
                # Note that E_{p(M|Z)}[-log s(M|\tilde{Z})]=KL[t(M|Z)|t(M|\tilde{Z})] + H[M|Z]
                # so minimizing the cross entropy is equivalent to minimizing the desired KL and increasing beta by 1
                # ie this also ensures invariance and determinism because the teacher is not detached. This is why
                # beta_det_inv is a much smaller value than lambda_maximality ( the effective beta is 1 + beta_det_inv).
                list_CE_pMlz_qMlza += [- (p_Mlz * log_q_Mlza).sum(-1).mean()]
            #########################

        CE_distill_aug = mean(list_CE_pMlz_qMlza)
        H_maximality = mean(list_H_M)
        CE_det_inv = mean(list_CE_pMlz_pMlza)

        # shape: []
        loss = CE_distill_aug - self.lambda_maximality * H_maximality + self.beta_det_inv * CE_det_inv

        # useful values to monitor
        H_Mlz = Categorical(probs=torch.cat(all_p_Mlz, dim=0).detach()).entropy().mean()
        logs = dict(
            H_M=H_maximality,
            H_Mlz=H_Mlz,
            KL_invariance=CE_det_inv - H_Mlz,
            KL_distillation=mean(list_CE_pMlz_qMlz) - H_Mlz,
            CE_distill_aug=CE_distill_aug
        )

        return loss, logs