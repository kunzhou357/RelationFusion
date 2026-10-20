"""Beta Prior Decision Learning and first-stage uncertainty calibration.

Paper: "Difference-aware Decision Learning for Multimodal Image Fusion"
(ICML 2026), Section 2.3 (Eqs. 14-21), Section 3.2 (Eqs. 31-34), and
Appendix D.2 (monotonicity of evidence insufficiency).

Scope: this module converts two non-negative evidence tensors into a Beta
prior over the contribution of modality B.  It does not compute features, FFT
statistics, image fusion or reconstruction.  It is intended to be called on
patch-level evidence ``[B, N]`` produced by the CDG evidence branch.

Equation-to-code mapping
------------------------
Eq. (15)  Z_B ~ Beta(e_B + lambda, e_A + lambda)
          -> ``alpha = e_B + lambda``, ``beta = e_A + lambda``
             (order is deliberate: alpha is the evidence of B).
Eq. (16)  W_B^raw = (e_B + lambda) / (e_A + e_B + 2 lambda)
          -> ``weight_raw``
Eq. (19)  U = 2 lambda / (e_A + e_B + 2 lambda)
          -> ``uncertainty``
Eq. (20)  Wcal = (1 - U) W_B^raw + 0.5 U
          -> ``weight_calibrated``   (FIRST calibration, inside this module)
Eq. (32)  patch-level form of the same first calibration
          -> identical expression in ``forward``
Eq. (34)  W_final = (1 - U_map) W_backfilled + 0.5 U_map
          -> ``BetaEvidenceDecision.calibrate_spatial``
             (SECOND calibration, called by CDG *after* fold-and-average)

The two calibrations are intentionally separate.  ``forward`` performs only
the first one on patch-level evidence.  ``calibrate_spatial`` is a static
utility for CDG; it must be applied to the backfilled weight and uncertainty
maps.  Do not merge or drop either calibration.

Numerical notes
---------------
* Evidence must be finite and non-negative; negative evidence raises a
  ``ValueError`` instead of being silently clamped.  This avoids masking
  upstream MDA/CDG bugs.
* ``alpha = e_B + lambda`` and ``beta = e_A + lambda`` are strictly positive
  because ``lambda > 0``.
* Ratios are evaluated in an overflow-safe scaled form: both evidences and
  ``lambda`` are divided by ``m = max(e_A, e_B, lambda) > 0``.  This is
  algebraically identical to the paper formulas, but avoids overflow of
  ``e_A + e_B + 2 lambda`` for very large finite float32 evidence.
* For float16/bfloat16 inputs, computation is promoted to float32 and the
  returned tensors are float32.  Gradients flow back through the promotion.
* No ``detach``, no in-place mutation of inputs, and no learnable parameters
  are used.  The Beta variance is returned only as a diagnostic; it is never
  used in place of the paper's uncertainty ``U``.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor, nn

__all__ = ["BetaEvidenceDecision", "promote_compute_dtype"]


def promote_compute_dtype(tensor: Tensor) -> torch.dtype:
    """Return a numerically safe dtype for the Beta computations."""
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return tensor.dtype


class BetaEvidenceDecision(nn.Module):
    r"""Beta prior decision with evidence-insufficiency calibration.

    Args:
        lambda_prior: Strictly positive pseudo-count ``lambda`` in Eq. (15).
            Appendix B fixes ``lambda = 1`` in the experiments; the default is
            therefore ``1.0``.

    Inputs:
        evidence_a: Non-negative finite evidence of modality A, ``[B, N]``.
        evidence_b: Non-negative finite evidence of modality B, same shape.

    Returns:
        A dictionary with tensors of the same shape as the inputs:

        ``alpha``
            First Beta shape parameter, ``e_B + lambda`` (Eq. 15).
        ``beta``
            Second Beta shape parameter, ``e_A + lambda`` (Eq. 15).
        ``weight_raw``
            Beta mean / raw contribution weight of modality B (Eq. 16).
        ``uncertainty``
            Evidence insufficiency ``U`` (Eq. 19).
        ``weight_calibrated``
            First-stage calibrated contribution weight of modality B
            (Eqs. 20/32).
        ``weight_a_raw``
            Raw contribution weight of modality A, ``1 - weight_raw``.
        ``weight_a_calibrated``
            First-stage calibrated weight of modality A,
            ``1 - weight_calibrated``.
        ``variance``
            Diagnostic Beta variance; **not** used as the uncertainty.

    All weights and ``uncertainty`` lie in ``[0, 1]`` by construction.
    """

    def __init__(self, lambda_prior: float = 1.0) -> None:
        super().__init__()
        lambda_prior = float(lambda_prior)
        if not math.isfinite(lambda_prior) or lambda_prior <= 0.0:
            raise ValueError(
                "lambda_prior must be a finite positive number, got "
                f"{lambda_prior}."
            )
        self.lambda_prior = lambda_prior

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_pair(self, evidence_a: Tensor, evidence_b: Tensor) -> None:
        for name, tensor in (
            ("evidence_a", evidence_a),
            ("evidence_b", evidence_b),
        ):
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if not tensor.dtype.is_floating_point:
                raise TypeError(
                    f"{name} must be a floating-point tensor, got {tensor.dtype}."
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN or Inf.")
            if (tensor < 0).any():
                raise ValueError(
                    f"{name} contains negative evidence. Evidence must be "
                    "non-negative; fix the upstream MDA/CDG computation rather "
                    "than clamping it silently."
                )
        if evidence_a.shape != evidence_b.shape:
            raise ValueError(
                "evidence_a and evidence_b must have the same shape, got "
                f"{tuple(evidence_a.shape)} and {tuple(evidence_b.shape)}."
            )
        if evidence_a.device != evidence_b.device:
            raise ValueError(
                "evidence_a and evidence_b must be on the same device, got "
                f"{evidence_a.device} and {evidence_b.device}."
            )
        if evidence_a.dtype != evidence_b.dtype:
            raise ValueError(
                "evidence_a and evidence_b must have the same dtype, got "
                f"{evidence_a.dtype} and {evidence_b.dtype}."
            )

    # ------------------------------------------------------------------
    # Forward: Beta prior + first calibration
    # ------------------------------------------------------------------

    def forward(self, evidence_a: Tensor, evidence_b: Tensor) -> Dict[str, Tensor]:
        self._validate_pair(evidence_a, evidence_b)

        compute_dtype = promote_compute_dtype(evidence_a)
        e_a = evidence_a.to(dtype=compute_dtype)
        e_b = evidence_b.to(dtype=compute_dtype)
        lam = torch.as_tensor(
            self.lambda_prior, dtype=compute_dtype, device=e_a.device
        )
        # A valid Python float need not be representable in float32.  Report
        # this explicitly instead of generating zero/inf pseudo-counts.
        if not torch.isfinite(lam) or lam <= 0:
            raise ValueError("lambda_prior is not representable in the compute dtype.")
        if ((e_a > torch.finfo(compute_dtype).max - lam).any()
                or (e_b > torch.finfo(compute_dtype).max - lam).any()):
            raise ValueError("Beta shape parameters overflow; use float64 evidence.")

        # Overflow-safe scaling.  m >= lambda > 0, so division is always safe.
        # The ratios below are algebraically identical to Eqs. (16) and (19).
        scale = torch.maximum(torch.maximum(e_a, e_b), lam)
        e_a_scaled = e_a / scale
        e_b_scaled = e_b / scale
        lam_scaled = lam / scale
        denominator_scaled = e_a_scaled + e_b_scaled + 2.0 * lam_scaled

        weight_raw = (e_b_scaled + lam_scaled) / denominator_scaled
        uncertainty = (2.0 * lam_scaled) / denominator_scaled

        # Eq. (20)/(32): first-stage uncertainty calibration.
        weight_calibrated = (
            (1.0 - uncertainty) * weight_raw + 0.5 * uncertainty
        )

        # Eq. (15): alpha belongs to B, beta belongs to A.  Do not swap.
        alpha = e_b + lam
        beta = e_a + lam

        # Diagnostic only: Beta variance, computed in a scaled form to avoid
        # overflow of alpha + beta for very large evidence.
        alpha_beta_scale = torch.maximum(alpha, beta)
        alpha_s = alpha / alpha_beta_scale
        beta_s = beta / alpha_beta_scale
        total_s = alpha_s + beta_s
        # Scaling by at least one also handles representable tiny priors:
        # directly taking 1/alpha_beta_scale can overflow when alpha,beta<1.
        variance_scale = alpha_beta_scale.clamp_min(1.0)
        inverse_scale = variance_scale.reciprocal()
        variance = (alpha_s * beta_s / total_s.square()) * (
            inverse_scale / (
                (alpha_beta_scale / variance_scale) * total_s + inverse_scale
            )
        )

        return {
            "alpha": alpha,
            "beta": beta,
            "weight_raw": weight_raw,
            "uncertainty": uncertainty,
            "weight_calibrated": weight_calibrated,
            "weight_a_raw": 1.0 - weight_raw,
            "weight_a_calibrated": 1.0 - weight_calibrated,
            "variance": variance,
        }

    # ------------------------------------------------------------------
    # Second calibration for CDG (post fold-and-average), Eq. (34)
    # ------------------------------------------------------------------

    @staticmethod
    def calibrate_spatial(
        weight_map: Tensor, uncertainty_map: Tensor
    ) -> Tensor:
        r"""Second-stage spatial calibration of Eq. (34).

        ``W_final = (1 - U_map) * W_backfilled + 0.5 * U_map``

        This is **not** the same step as the first calibration performed in
        :meth:`forward`.  It is intended to be called by CDG *after* the
        patch weights and uncertainties have been backfilled into spatial maps
        by the fold-and-average operator of Eq. (33).

        Args:
            weight_map: Backfilled patch-weight map, values in ``[0, 1]``.
            uncertainty_map: Backfilled uncertainty map, values in ``[0, 1]``.
                Any shape is accepted (e.g. ``[B, N]`` or ``[B, 1, H, W]``).

        Returns:
            Calibrated map in ``[0, 1]`` with the same shape.  The returned
            dtype is float32 for float16/bfloat16 inputs, otherwise the input
            dtype.
        """
        for name, tensor in (
            ("weight_map", weight_map),
            ("uncertainty_map", uncertainty_map),
        ):
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if not tensor.dtype.is_floating_point:
                raise TypeError(
                    f"{name} must be a floating-point tensor, got {tensor.dtype}."
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN or Inf.")
        if weight_map.shape != uncertainty_map.shape:
            raise ValueError(
                "weight_map and uncertainty_map must have the same shape, got "
                f"{tuple(weight_map.shape)} and {tuple(uncertainty_map.shape)}."
            )
        if weight_map.device != uncertainty_map.device:
            raise ValueError(
                "weight_map and uncertainty_map must be on the same device."
            )
        if weight_map.dtype != uncertainty_map.dtype:
            raise ValueError(
                "weight_map and uncertainty_map must have the same dtype, got "
                f"{weight_map.dtype} and {uncertainty_map.dtype}."
            )
        if (weight_map < 0).any() or (weight_map > 1).any():
            raise ValueError(
                "weight_map values must lie in [0, 1]. Fix the upstream "
                "fold-and-average result rather than clipping silently."
            )
        if (uncertainty_map < 0).any() or (uncertainty_map > 1).any():
            raise ValueError(
                "uncertainty_map values must lie in [0, 1]. Fix the upstream "
                "fold-and-average result rather than clipping silently."
            )

        compute_dtype = promote_compute_dtype(weight_map)
        weights = weight_map.to(dtype=compute_dtype)
        uncertainty = uncertainty_map.to(dtype=compute_dtype)
        return (1.0 - uncertainty) * weights + 0.5 * uncertainty

    def extra_repr(self) -> str:
        return f"lambda_prior={self.lambda_prior}"
