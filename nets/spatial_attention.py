"""Modal Difference Attention (MDA) for IDEAL.

Paper: "Difference-aware Decision Learning for Multimodal Image Fusion"
(ICML 2026), Figure 2, Section 3.3, Eq. (36), and Appendix A/Table 5.

Scope: MDA only.  The Conditional Decision Gating (CDG) module, the Beta
decision policy, the encoder/decoder and the complete fusion network are not
implemented here.

What the paper specifies
------------------------
* The cross-modal spatial feature difference is ``D = F_A - F_B`` (Figure 2
  shows the subtraction; Appendix A defines ``D^l`` as "cross-modal spatial
  feature difference").
* ``Q``, ``K`` and ``V`` are obtained through three different projection
  convolutions (Section 3.3); Appendix A states that the projections are
  implemented by 1x1 convolutions.
* ``A_context = softmax(Q K^T / sqrt(d)) V`` (Eq. 36).
* ``d`` is the head dimension used as the attention scaling factor
  (Appendix A: ``d_h``).
* The contextual representation is upsampled with a 1x1 convolution kernel to
  restore the difference map to the feature-map size (Section 3.3).
* The final attention map is the single-channel map ``A`` consumed by CDG's
  patch average ``d_n`` in Eq. (29).

What the paper does not specify (see ``docs/design.md``)
------------------------------------------------------------------
* the attention axis and tokenization / pooling strategy;
* the embedding width, the upsampling mode and the final 1x1 convolution;
* whether the difference is signed or absolute (signed is the default here);
* the exact roles of the six 1x1 projections ``phi_q, phi_k, phi_v, phi_o,
  phi_d, phi_m`` listed in Appendix A (only q/k/v are unambiguous);
* initialization details;
* the final Sigmoid.  Appendix A lists ``sigma`` as the function used to
  generate the final attention map, but Eq. (36) itself does not contain it;
  in this reproduction the Sigmoid is applied to satisfy the non-negativity
  requirement of the downstream CDG evidence, and is explicitly documented as
  an assumption.

No residual connection, LayerNorm, multi-head attention, extra gating,
Transformer/Mamba block or enhancement network is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from utils.numerics import align_autocast_inputs, validate_tensor

__all__ = ["MDA", "MDADebug", "scaled_dot_product_attention"]


def _adaptive_pool_exact(x, output_size):
    """Exact adaptive-average bins without MPS's divisibility restriction.

    Separable averaging matrices implement PyTorch's floor/ceil bin boundaries.
    This fallback is used only for non-divisible MPS shapes, never a CPU copy.
    """
    height, width = x.shape[-2:]
    out_h, out_w = output_size
    dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    weights = []
    for source, target in ((height, out_h), (width, out_w)):
        index = torch.arange(target, device=x.device)
        start = torch.div(index * source, target, rounding_mode="floor")
        stop = torch.div((index + 1) * source + target - 1, target, rounding_mode="floor")
        positions = torch.arange(source, device=x.device)
        mask = (positions[None, :] >= start[:, None]) & (positions[None, :] < stop[:, None])
        weights.append(mask.to(dtype) / (stop - start).to(dtype)[:, None])
    with torch.autocast(device_type=x.device.type, enabled=False):
        result = weights[0] @ x.to(dtype) @ weights[1].transpose(0, 1)
    return result.to(x.dtype)


def _adaptive_pool(x, size):
    if x.device.type == "mps" and (x.shape[-2] % size[0] or x.shape[-1] % size[1]):
        return _adaptive_pool_exact(x, size)
    return F.adaptive_avg_pool2d(x, size)


# ---------------------------------------------------------------------------
# Stable single-head scaled dot-product attention
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, scale: float
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Compute ``softmax(Q K^T * scale) V`` for a single head.

    Args:
        q: Queries ``[B, N, D]``.
        k: Keys ``[B, M, D]``.
        v: Values ``[B, M, Dv]``.
        scale: Multiplicative scale applied to the logits, typically
            ``1 / sqrt(D)``.  ``D`` is the embedding/head dimension, not the
            number of spatial tokens.

    Returns:
        ``(logits, attention_weights, context)`` with shapes
        ``[B, N, M]``, ``[B, N, M]`` and ``[B, N, Dv]``.
        The Softmax is taken along the key/token dimension (``dim=-1``).

    Numerical notes:
        For float16/bfloat16 inputs the matrix products and the Softmax are
        promoted to float32 and the context is cast back to the input dtype.
        This is an implementation detail for AMP/FP16 stability; it does not
        change the equation and does not detach the graph.
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            "q, k and v must be 3-D tensors [B, N, D] / [B, M, D] / "
            "[B, M, Dv]."
        )
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k and v must share the batch dimension.")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            "q and k must have the same embedding dimension, got "
            f"{q.shape[-1]} and {k.shape[-1]}."
        )
    if k.shape[1] != v.shape[1]:
        raise ValueError(
            "k and v must have the same number of tokens, got "
            f"{k.shape[1]} and {v.shape[1]}."
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k and v must have the same dtype.")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale}.")

    compute_dtype = (
        torch.float32
        if q.dtype in (torch.float16, torch.bfloat16)
        else q.dtype
    )
    q_compute = q.to(dtype=compute_dtype)
    k_compute = k.to(dtype=compute_dtype)
    v_compute = v.to(dtype=compute_dtype)

    # Matrix multiplication, never element-wise multiplication.
    # Explicit casts alone do not disable an enclosing autocast context:
    # autocast would cast these matmuls back to fp16/bf16.
    with torch.autocast(device_type=q.device.type, enabled=False):
        logits = torch.matmul(q_compute, k_compute.transpose(-2, -1)) * float(scale)
        attention_weights = torch.softmax(logits, dim=-1)
        context = torch.matmul(attention_weights, v_compute)
    return logits, attention_weights, context.to(dtype=q.dtype)


# ---------------------------------------------------------------------------
# Debug container
# ---------------------------------------------------------------------------


def _tensor_stats(tensor: Tensor) -> Dict[str, float]:
    """Cheap finite statistics of a tensor (detached, no graph retained)."""
    with torch.no_grad():
        value = tensor.detach()
        if value.is_complex():
            value = value.abs()
        value = value.float()
        if value.numel() == 0:
            return {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "absmax": float("nan"),
            }
        std = value.std().item() if value.numel() > 1 else 0.0
        return {
            "min": value.min().item(),
            "max": value.max().item(),
            "mean": value.mean().item(),
            "std": std,
            "absmax": value.abs().max().item(),
        }


@dataclass
class MDADebug:
    """Shapes and statistics of the intermediate MDA tensors.

    By default no large tensor is stored.  If the module was constructed with
    ``store_debug_tensors=True``, ``tensors`` contains detached clones of all
    intermediates (including the attention matrix, whose size is bounded by
    ``max_tokens`` in pooled-spatial mode).

    Attributes:
        metadata: Attention mode, token count, pooling grid, embedding width,
            scale, input shape and related values.
        shapes: Mapping from intermediate name to shape tuple.
        stats: Mapping from intermediate name to ``min/max/mean/std/absmax``.
        tensors: Optional mapping from intermediate name to a detached tensor.
    """

    metadata: Dict[str, Any]
    shapes: Dict[str, Tuple[int, ...]]
    stats: Dict[str, Dict[str, float]]
    tensors: Optional[Dict[str, Tensor]] = None

    def summary(self) -> str:
        lines = ["MDA debug summary:"]
        for key, shape in self.shapes.items():
            stat = self.stats[key]
            lines.append(
                f"  {key:<20} shape={tuple(shape)} "
                f"min={stat['min']:.4e} max={stat['max']:.4e} "
                f"mean={stat['mean']:.4e} std={stat['std']:.4e}"
            )
        for key, value in self.metadata.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MDA
# ---------------------------------------------------------------------------


class MDA(nn.Module):
    r"""Modal Difference Attention.

    Pipeline (pooled-spatial mode, the default):

    1. ``D = F_A - F_B`` (optionally ``|F_A - F_B|`` for ablation).
    2. ``Q_map = phi_q(D)``, ``K_map = phi_k(D)``, ``V_map = phi_v(D)`` with
       three independent 1x1 convolutions.
    3. Adaptive average pooling of the Q/K/V maps to at most ``max_tokens``
       spatial tokens; flatten to ``[B, N, D]``.
    4. ``logits = Q K^T / sqrt(D)``; ``attention = softmax(logits, dim=-1)``;
       ``context = attention V``.
    5. Reshape the context to ``[B, D, H_p, W_p]`` and upsample it to
       ``[B, D, H, W]``.
    6. A 1x1 convolution projects it to one channel, followed by Sigmoid:
       ``A = sigmoid(Conv1x1(upsample(context)))``, shape ``[B, 1, H, W]``.

    ``attention_mode='channel'`` is an alternative comparison implementation
    (channel tokens instead of pooled spatial tokens).  It is **not** claimed
    to be the author's choice.

    Args:
        channels: Number of channels ``C`` of both input features.
        embed_dim: Width of the Q/K/V projections, used as the head dimension
            ``d`` in the attention scale ``1 / sqrt(d)``.  Defaults to
            ``channels``.  The paper does not report its value.
        max_tokens: Upper bound on the number of spatial tokens in
            pooled-spatial mode.  Defaults to ``256``; this is an engineering
            assumption and avoids an ``O((HW)^2)`` attention matrix.
        attention_mode: ``'pooled_spatial'`` (default) or ``'channel'``.
        use_abs_difference: If ``True``, use ``|F_A - F_B|``.  The paper does
            not state whether the difference is absolute; the default
            (``False``) is the signed difference, and the absolute variant is
            an ablation only.
        upsample_mode: Interpolation mode used to restore the pooled context to
            the input resolution.  One of ``'bilinear'`` (default),
            ``'nearest'``, ``'bicubic'``.  The paper only says "upsampling".
        store_debug_tensors: If ``True``, debug calls additionally return
            detached clones of all intermediate tensors.  Defaults to
            ``False`` so that a giant attention matrix is not retained.

    Inputs:
        feature_a: First modality feature ``[B, C, H, W]``.
        feature_b: Second modality feature ``[B, C, H, W]``; same shape,
            scale, device and dtype as ``feature_a``.

    Returns:
        ``A`` of shape ``[B, 1, H, W]`` when ``return_debug=False``; otherwise
        ``(A, MDADebug)``.  ``A`` is a spatial difference attention map intended
        for the CDG patch average, not a final modality-fusion weight.
    """

    _ATTENTION_MODES = ("pooled_spatial", "channel")
    _UPSAMPLE_MODES = ("bilinear", "nearest", "bicubic")

    def __init__(
        self,
        channels: int,
        embed_dim: Optional[int] = None,
        max_tokens: int = 256,
        attention_mode: str = "pooled_spatial",
        use_abs_difference: bool = False,
        upsample_mode: str = "bilinear",
        store_debug_tensors: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError(
                f"channels must be a positive integer, got {channels!r}."
            )
        if embed_dim is None:
            embed_dim = channels
        if not isinstance(embed_dim, int) or embed_dim < 1:
            raise ValueError(
                f"embed_dim must be a positive integer or None, got {embed_dim!r}."
            )
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(
                f"max_tokens must be a positive integer, got {max_tokens!r}."
            )
        if attention_mode not in self._ATTENTION_MODES:
            raise ValueError(
                f"attention_mode must be one of {self._ATTENTION_MODES}, got "
                f"{attention_mode!r}."
            )
        if upsample_mode not in self._UPSAMPLE_MODES:
            raise ValueError(
                f"upsample_mode must be one of {self._UPSAMPLE_MODES}, got "
                f"{upsample_mode!r}."
            )
        if not isinstance(use_abs_difference, bool):
            raise TypeError(
                "use_abs_difference must be a bool, got "
                f"{type(use_abs_difference)!r}."
            )

        self.channels = channels
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens
        self.attention_mode = attention_mode
        self.use_abs_difference = use_abs_difference
        self.upsample_mode = upsample_mode
        self.store_debug_tensors = bool(store_debug_tensors)

        # Three independent 1x1 projections of the difference, plus the 1x1
        # output projection to a single-channel map.  Default PyTorch init.
        self.q_proj = nn.Conv2d(channels, embed_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, embed_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, embed_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(embed_dim, 1, kernel_size=1)

    # ------------------------------------------------------------------
    # Validation and token-grid helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, feature_a: Tensor, feature_b: Tensor) -> None:
        validate_tensor("feature_a", feature_a)
        validate_tensor("feature_b", feature_b)
        for name, tensor in (("feature_a", feature_a), ("feature_b", feature_b)):
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if tensor.dim() != 4:
                raise ValueError(
                    f"{name} must have shape [B, C, H, W], got "
                    f"{tuple(tensor.shape)}."
                )
            if not tensor.dtype.is_floating_point:
                raise TypeError(
                    f"{name} must be a floating-point tensor, got {tensor.dtype}."
                )
        if feature_a.shape != feature_b.shape:
            raise ValueError(
                "feature_a and feature_b must have the same shape, got "
                f"{tuple(feature_a.shape)} and {tuple(feature_b.shape)}."
            )
        if feature_a.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} input channels, got "
                f"{feature_a.shape[1]}."
            )
        if feature_a.device != feature_b.device:
            raise ValueError(
                "feature_a and feature_b must be on the same device, got "
                f"{feature_a.device} and {feature_b.device}."
            )
        if feature_a.dtype != feature_b.dtype:
            raise ValueError(
                "feature_a and feature_b must have the same dtype, got "
                f"{feature_a.dtype} and {feature_b.dtype}."
            )
        parameter = next(self.parameters())
        if parameter.device != feature_a.device:
            raise ValueError(
                f"Input device {feature_a.device} does not match module device "
                f"{parameter.device}."
            )
        if parameter.dtype != feature_a.dtype:
            raise ValueError(
                f"Input dtype {feature_a.dtype} does not match module dtype "
                f"{parameter.dtype}."
            )

    @staticmethod
    def _pool_grid(height: int, width: int, max_tokens: int) -> Tuple[int, int]:
        """Choose a proportional spatial grid with at most ``max_tokens`` cells."""
        if height * width <= max_tokens:
            return height, width

        aspect = width / height
        pool_h = max(1, int(math.floor(math.sqrt(max_tokens / aspect))))
        pool_w = max(1, max_tokens // pool_h)
        while pool_h * pool_w > max_tokens:
            if pool_h >= pool_w:
                pool_h = max(1, pool_h - 1)
            else:
                pool_w = max(1, pool_w - 1)

        pool_h = min(pool_h, height)
        pool_w = min(pool_w, width)
        return pool_h, pool_w

    def _interpolate_context(self, context: Tensor, size: Tuple[int, int]) -> Tensor:
        if context.shape[-2:] == size:
            return context
        kwargs = {} if self.upsample_mode == "nearest" else {"align_corners": False}
        return F.interpolate(
            context, size=size, mode=self.upsample_mode, **kwargs
        )

    # ------------------------------------------------------------------
    # Debug construction
    # ------------------------------------------------------------------

    def _build_debug(
        self,
        named_tensors: Dict[str, Tensor],
        metadata: Dict[str, Any],
    ) -> MDADebug:
        shapes = {
            name: tuple(tensor.shape)
            for name, tensor in named_tensors.items()
        }
        stats = {
            name: _tensor_stats(tensor)
            for name, tensor in named_tensors.items()
        }
        tensors = None
        if self.store_debug_tensors:
            tensors = {
                name: tensor.detach().clone()
                for name, tensor in named_tensors.items()
            }
        return MDADebug(
            metadata=metadata, shapes=shapes, stats=stats, tensors=tensors
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        feature_a: Tensor,
        feature_b: Tensor,
        return_debug: bool = False,
    ):
        feature_a, feature_b = align_autocast_inputs(self, feature_a, feature_b)
        self._validate_inputs(feature_a, feature_b)
        batch, _, height, width = feature_a.shape

        # Cross-modal spatial difference.  Signed by default; the absolute
        # value is an optional ablation, not a paper-specified operation.
        difference = feature_a - feature_b
        if self.use_abs_difference:
            difference = difference.abs()

        # Three independent 1x1 projections (Section 3.3, Appendix A).
        q_map = self.q_proj(difference)
        k_map = self.k_proj(difference)
        v_map = self.v_proj(difference)

        if self.attention_mode == "pooled_spatial":
            pool_h, pool_w = self._pool_grid(height, width, self.max_tokens)
            if (pool_h, pool_w) != (height, width):
                q_tokens = (
                    _adaptive_pool(q_map, (pool_h, pool_w))
                    .flatten(2)
                    .transpose(1, 2)
                )
                k_tokens = (
                    _adaptive_pool(k_map, (pool_h, pool_w))
                    .flatten(2)
                    .transpose(1, 2)
                )
                v_tokens = (
                    _adaptive_pool(v_map, (pool_h, pool_w))
                    .flatten(2)
                    .transpose(1, 2)
                )
            else:
                q_tokens = q_map.flatten(2).transpose(1, 2)
                k_tokens = k_map.flatten(2).transpose(1, 2)
                v_tokens = v_map.flatten(2).transpose(1, 2)

            num_tokens = pool_h * pool_w
            # d is the embedding/head dimension, NOT the token count.
            scale = 1.0 / math.sqrt(self.embed_dim)
            logits, attention_weights, context = scaled_dot_product_attention(
                q_tokens, k_tokens, v_tokens, scale
            )
            context_map = context.transpose(1, 2).reshape(
                batch, self.embed_dim, pool_h, pool_w
            )
            upsampled_context = self._interpolate_context(
                context_map, (height, width)
            )
        else:
            # Alternative channel-attention comparison.  Tokens are channels;
            # the dot-product dimension is the number of spatial positions.
            q_tokens = q_map.flatten(2)  # [B, D, H*W]
            k_tokens = k_map.flatten(2)
            v_tokens = v_map.flatten(2)
            num_tokens = self.embed_dim
            pool_h, pool_w = height, width
            scale = 1.0 / math.sqrt(height * width)
            logits, attention_weights, context = scaled_dot_product_attention(
                q_tokens, k_tokens, v_tokens, scale
            )
            context_map = context.reshape(batch, self.embed_dim, height, width)
            upsampled_context = context_map

        # Single-channel spatial map.  Sigmoid is an explicit reproduction
        # assumption (see the module docstring and assumptions file).
        map_logits = self.out_proj(upsampled_context)
        attention_map = torch.sigmoid(map_logits)

        if not isinstance(return_debug, bool):
            raise TypeError(
                f"return_debug must be a bool, got {type(return_debug)!r}."
            )
        if not return_debug:
            return attention_map

        debug = self._build_debug(
            named_tensors={
                "difference": difference,
                "Q": q_tokens,
                "K": k_tokens,
                "V": v_tokens,
                "attention_logits": logits,
                "attention_weights": attention_weights,
                "context": context,
                "context_map": context_map,
                "upsampled_context": upsampled_context,
                "attention_map": attention_map,
            },
            metadata={
                "attention_mode": self.attention_mode,
                "embed_dim": self.embed_dim,
                "input_shape": (batch, self.channels, height, width),
                "num_tokens": num_tokens,
                "pool_grid": (pool_h, pool_w),
                "scale": scale,
                "max_tokens": self.max_tokens,
                "use_abs_difference": self.use_abs_difference,
                "upsample_mode": self.upsample_mode,
            },
        )
        return attention_map, debug

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, embed_dim={self.embed_dim}, "
            f"max_tokens={self.max_tokens}, "
            f"attention_mode={self.attention_mode!r}, "
            f"use_abs_difference={self.use_abs_difference}, "
            f"upsample_mode={self.upsample_mode!r}"
        )
