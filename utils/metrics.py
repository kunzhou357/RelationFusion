"""Luminance metrics with explicit undefined cases and documented conventions."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from .losses import StructuralSimilarity
from .operators import Sobel


def _correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a*b).sum() / denominator) if denominator > 1e-12 else None


def _mutual_information(a, b):
    hist = np.histogram2d(a.ravel(), b.ravel(), bins=256, range=((0, 256), (0, 256)))[0]
    joint = hist / hist.sum()
    marginal = joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)
    mask = joint > 0
    return float((joint[mask] * np.log2(joint[mask] / marginal[mask])).sum())


def _vif(reference, fused):
    if float(np.var(reference)) < 1e-12:
        return None
    reference = torch.tensor(reference, dtype=torch.float64)[None, None]
    fused = torch.tensor(fused, dtype=torch.float64)[None, None]
    numerator, denominator = 0., 0.
    for scale in range(1, 5):
        size = 2**(5-scale) + 1
        coordinate = torch.arange(size, dtype=torch.float64) - size//2
        gaussian = torch.exp(-coordinate.square()/(2*(size/5)**2))
        kernel = (gaussian[:, None] * gaussian[None, :])
        kernel = (kernel/kernel.sum())[None, None]
        if min(reference.shape[-2:]) < size:
            return None
        if scale > 1:
            reference, fused = [F.conv2d(x, kernel)[..., ::2, ::2] for x in (reference, fused)]
        if min(reference.shape[-2:]) < size:
            return None
        average = lambda x: F.conv2d(x, kernel)
        m1, m2 = average(reference), average(fused)
        v1 = (average(reference.square()) - m1.square()).clamp_min(0)
        v2 = (average(fused.square()) - m2.square()).clamp_min(0)
        cov = average(reference*fused) - m1*m2
        gain = cov / (v1 + 1e-10)
        noise = v2 - gain*cov
        noise = torch.where(v1 < 1e-10, v2, noise)
        gain = torch.where((v1 < 1e-10) | (v2 < 1e-10), torch.zeros_like(gain), gain)
        noise = torch.where(v2 < 1e-10, torch.zeros_like(noise), noise)
        noise = torch.where(gain < 0, v2, noise).clamp_min(1e-10)
        gain = gain.clamp_min(0)
        numerator += float(torch.log1p(gain.square()*v1/(noise+2.)).sum())
        denominator += float(torch.log1p(v1/2.).sum())
    return numerator/denominator if denominator > 1e-12 else None


def _qabf(ir, vi, fused):
    sobel = Sobel()
    gradients = [sobel(torch.tensor(x, dtype=torch.float32)[None, None])[0, 0] for x in (ir, vi, fused)]
    magnitudes = [torch.linalg.vector_norm(g, dim=0) for g in gradients]
    angles = [torch.atan2(g[1], g[0]) for g in gradients]
    denominator = float((magnitudes[0]+magnitudes[1]).sum())
    if denominator <= 1e-12:
        return None
    total = 0.
    for index in (0, 1):
        strength = torch.minimum(magnitudes[index], magnitudes[2]) / torch.maximum(
            magnitudes[index], magnitudes[2]).clamp_min(1e-12)
        angle = (angles[index] - angles[2]).abs().remainder(math.pi)
        orientation = 1 - torch.minimum(angle, math.pi-angle)/(math.pi/2)
        quality = (.9994 * torch.sigmoid(15*(strength-.5))) * (.9879 * torch.sigmoid(22*(orientation-.8)))
        total += float((quality * magnitudes[index]).sum())
    return total/denominator


@torch.no_grad()
def fusion_metrics(fused, ir, vi):
    tensors = [x.detach().float().cpu().reshape(1, 1, *x.shape[-2:]) for x in (fused, ir, vi)]
    if len({tuple(t.shape) for t in tensors}) != 1 or min(tensors[0].shape[-2:]) < 2:
        raise ValueError("Metrics require matching images with height/width >=2")
    if any(not torch.isfinite(t).all() for t in tensors):
        raise ValueError("Metrics require finite image values")
    arrays = [np.rint(np.clip(t.numpy()[0, 0], 0, 1)*255).astype(np.float64) for t in tensors]
    tensors = [torch.from_numpy(a.astype(np.float32)/255)[None, None] for a in arrays]
    f, a, b = arrays
    probability = np.bincount(f.astype(np.uint8).ravel(), minlength=256) / f.size
    positive = probability > 0
    dy, dx = np.gradient(f)
    c1, c2 = _correlation(f-a, b), _correlation(f-b, a)
    cc1, cc2 = _correlation(f, a), _correlation(f, b)
    vif1, vif2 = _vif(a, f), _vif(b, f)
    ssim = StructuralSimilarity()
    result = {"EN": float(-(probability[positive]*np.log2(probability[positive])).sum()),
              "SD": float(f.std()),
              "SF": float(np.sqrt(np.mean(np.diff(f, axis=0)**2)+np.mean(np.diff(f, axis=1)**2))),
              "AG": float(np.sqrt((dx*dx+dy*dy)/2).mean()),
              "MI": _mutual_information(f, a)+_mutual_information(f, b),
              "CC": (cc1+cc2)/2 if cc1 is not None and cc2 is not None else None,
              "SCD": c1+c2 if c1 is not None and c2 is not None else None,
              "VIF": vif1+vif2 if vif1 is not None and vif2 is not None else None,
              "Qabf": _qabf(a, b, f),
              "SSIM": float((ssim(tensors[0], tensors[1])+ssim(tensors[0], tensors[2]))*.5)}
    return {key: value if value is None or math.isfinite(value) else None for key, value in result.items()}
