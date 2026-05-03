"""
Regularizers inspired by LeWorldModel (Maes, Le Lidec, Scieur, LeCun, Balestriero 2026)
and the precursor LeJEPA work (Balestriero & LeCun 2025).

We implement two complementary terms:

SIGReg — Sketched-Isotropic-Gaussian Regularizer
    Encourages the latent embedding distribution P_Z to match an isotropic
    Gaussian N(0, I) via the Cramér-Wold theorem: it suffices to match every
    one-dimensional marginal. SIGReg randomly projects Z onto M unit-norm
    directions u^(m) in S^{D-1} and computes the Epps-Pulley univariate
    normality test statistic T on each projection, averaged over m.

        SIGReg(Z) = (1/M) * Σ_m T(Z u^(m))

    T(h) integrates |empirical CF(t) - Gaussian CF(t)|^2 weighted by w(t),
    using trapezoidal quadrature over t ∈ [0.2, 4.0]. As SIGReg -> 0, P_Z -> N(0, I).

    This is a *principled* anti-collapse regularizer, replacing heuristics
    like EMA / stop-gradient with a single tunable hyperparameter (the weight λ).

ActionOrthogonality — intervention embedding spread
    Our observed failure mode: the three intervention embeddings (metformin,
    atorvastatin, lisinopril) learned by `nn.Embedding(K, z_dim)` cluster
    together, so `Predictor(s_x, z=lisinopril)` collapses toward
    `Predictor(s_x, z=atorvastatin)`. Direct fix: penalize off-diagonal entries
    of A A^T where A is the (K, z_dim) embedding matrix (centered and L2-norm
    rows). Pushes different interventions toward orthogonal directions in z-space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ======================================================================
# SIGReg
# ======================================================================

def sigreg(
    Z: torch.Tensor,
    num_projections: int = 1024,
    num_knots: int = 17,
    t_min: float = 0.2,
    t_max: float = 4.0,
    weight_sigma: float = 1.0,
    center_and_scale: bool = True,
) -> torch.Tensor:
    """Sketched-Isotropic-Gaussian Regularizer (SIGReg).

    Parameters
    ----------
    Z : (B, D) embeddings. Gradient flows through Z.
    num_projections (M) : Random unit directions. Paper uses 1024.
    num_knots (K) : Trapezoidal integration knots in [t_min, t_max]. Paper uses 17.
    t_min, t_max : Integration range. Paper default [0.2, 4.0].
    weight_sigma : Scale of the Gaussian weighting w(t) = exp(-t^2 / (2*sigma^2)).
    center_and_scale : If True, batch-normalize Z before SIGReg. This matches the
                       paper's approach of placing a BatchNorm right before the
                       anti-collapse objective so centering/scaling are not what
                       the term has to fight.

    Returns
    -------
    Scalar non-negative loss. 0 implies (asymptotically) P_Z = N(0, I).
    """
    B, D = Z.shape
    if B < 4:
        # Too few samples for a meaningful characteristic-function estimate.
        return Z.new_zeros(())

    # Normalize before matching to N(0, I). Gradient flows through the
    # (mean, var) because we use the online batch stats exactly like BN.
    if center_and_scale:
        Zc = Z - Z.mean(dim=0, keepdim=True)
        Zc = Zc / (Zc.std(dim=0, keepdim=True, unbiased=False) + 1e-5)
    else:
        Zc = Z

    # Random unit-norm directions on S^{D-1}
    U = torch.randn(num_projections, D, device=Z.device, dtype=Z.dtype)
    U = U / (U.norm(dim=-1, keepdim=True) + 1e-12)

    # 1D projections: H[b, m] = Zc[b] · U[m]
    H = Zc @ U.T                                  # (B, M)

    # Quadrature knots
    t = torch.linspace(t_min, t_max, num_knots, device=Z.device, dtype=Z.dtype)  # (K,)

    # Empirical characteristic function along each projection:
    #   ϕ_emp(t, m) = (1/B) Σ_b exp(i t H[b, m])
    # We compute Re and Im parts separately.
    tH = H.unsqueeze(-1) * t.view(1, 1, -1)       # (B, M, K)
    re_emp = torch.cos(tH).mean(dim=0)            # (M, K)
    im_emp = torch.sin(tH).mean(dim=0)            # (M, K)

    # Target CF for standard Gaussian N(0,1): ϕ_0(t) = exp(-t^2 / 2)
    re_tgt = torch.exp(-0.5 * t * t)              # (K,)
    # im_tgt = 0

    diff_sq = (re_emp - re_tgt.view(1, -1)) ** 2 + im_emp ** 2     # (M, K)

    # Weighting w(t) = exp(-t^2 / (2 σ^2))
    w = torch.exp(-0.5 * (t * t) / (weight_sigma * weight_sigma))  # (K,)
    integrand = diff_sq * w.view(1, -1)                            # (M, K)

    # Trapezoidal rule over t
    dt = (t_max - t_min) / (num_knots - 1)
    # sum = Δt * ( (f0 + fK-1)/2 + Σ_inner )
    trapz = dt * (integrand.sum(dim=-1)
                  - 0.5 * (integrand[:, 0] + integrand[:, -1]))      # (M,)

    return trapz.mean()


# ======================================================================
# Action-embedding orthogonality
# ======================================================================

def action_orthogonality_loss(
    action_embedding: nn.Embedding,
    target_offdiag: float = 0.0,
) -> torch.Tensor:
    """Encourage rows of ``action_embedding.weight`` to be mutually orthogonal
    unit vectors.

    Let A be the (K, z_dim) embedding matrix with K=3 interventions.
    Compute N = normalize_rows(A) and loss = || N N^T - I ||^2_F / K^2.

    Since K=3 and z_dim=16, orthogonal rows fit easily and the predictor
    gets clearly distinguishable conditioning vectors for each drug.
    """
    A = action_embedding.weight                               # (K, z_dim)
    K = A.shape[0]
    An = A / (A.norm(dim=-1, keepdim=True) + 1e-12)
    gram = An @ An.T                                          # (K, K)
    target = torch.eye(K, device=A.device, dtype=A.dtype) * (1.0 - target_offdiag) \
             + target_offdiag
    return ((gram - target) ** 2).sum() / (K * K)
