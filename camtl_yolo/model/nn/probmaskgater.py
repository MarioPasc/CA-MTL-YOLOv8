# mga_yolo/nn/modules/probmaskgater.py

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Literal

class ProbMaskGater(nn.Module):
    """
    Differentiable spatial gate for mask-guided attention.

    Modes
    -----
    - 'deterministic': M = p                    # más estable
    - 'gumbel':       M = sigmoid((logit(p)+g)/tau)
    - 'hard_st':      forward: (M>th).float(); backward: grad(M_soft) (Straight-Through)
    - 'bernoulli_detach': M = Bernoulli(p.detach())  # sin acoplar det→seg

    Args
    ----
    mode : {'deterministic','gumbel','hard_st','bernoulli_detach'}
    tau : float > 0, menor ⇒ puerta más binaria (para gumbel/hard_st)
    p_min : float ∈ [0,1], piso para evitar apagados totales
    threshold : float ∈ (0,1), umbral de la versión hard
    seed : Optional[int], para reproducibilidad del muestreo
    """
    def __init__(self,
                 mode: str = 'gumbel',
                 tau: float = 1.0,
                 p_min: float = 0.0,
                 threshold: float = 0.5,
                 seed: Optional[int] = None):
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.mode = mode
        self.tau = float(tau)
        self.p_min = float(p_min)
        self.threshold = float(threshold)
        self._seed = seed
        if seed is not None:
            self.register_buffer("_ctr", torch.zeros((), dtype=torch.long))

    def _rng(self, device):
        if self._seed is None:
            return None
        g = torch.Generator(device=device)
        g.manual_seed(self._seed + int(self._ctr.item()))
        self._ctr.add_(1)
        return g

    # Let's see when does PyTorch add this method natively... 
    # https://github.com/pytorch/pytorch/issues/101974
    @staticmethod
    def _rand_like(x: torch.Tensor, generator: torch.Generator | None):
        # preserves shape/dtype/device; uses local Generator if provided
        return torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=generator)


    @staticmethod
    def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def _gumbel_sigmoid_from_prob(self, p: torch.Tensor) -> torch.Tensor:
        gen = self._rng(p.device)
        U1 = self._rand_like(p, gen).clamp_(1e-6, 1 - 1e-6)
        U2 = self._rand_like(p, gen).clamp_(1e-6, 1 - 1e-6)
        g = -torch.log(-torch.log(U1)) - (-torch.log(-torch.log(U2)))  # Logistic noise
        logits = self._logit(p)
        return torch.sigmoid((logits + g) / self.tau)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        # p: (N,1,H,W) o (N,H,W) en [0,1]
        if p.dim() == 3:
            p = p.unsqueeze(1)
        p = p.float().clamp(0.0, 1.0)
        if self.p_min > 0:
            p = torch.maximum(p, torch.tensor(self.p_min, device=p.device, dtype=p.dtype))

        # durante eval, todo determinista
        if not self.training or self.mode == 'deterministic':
            return p

        if self.mode == 'gumbel':
            m_soft = self._gumbel_sigmoid_from_prob(p)
            return m_soft

        if self.mode == 'hard_st':
            m_soft = self._gumbel_sigmoid_from_prob(p)
            m_hard = (m_soft > self.threshold).float()
            return m_hard + (m_soft - m_soft.detach())

        if self.mode == 'bernoulli_detach':
            return torch.bernoulli(p.detach(), generator=self._rng(p.device))

        # fallback
        return p