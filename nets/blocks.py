"""Restormer-style channel attention and gated convolution building blocks."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = x.float() if x.dtype != torch.float64 else x
            var, mean = torch.var_mean(work, dim=1, keepdim=True, unbiased=False)
            return (work - mean) * torch.rsqrt(var + 1e-5) * self.weight + self.bias


def channel_attention(q, k, v, heads, temperature):
    b, c, h, w = q.shape
    with torch.autocast(device_type=q.device.type, enabled=False):
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        q, k, v = [x.to(dtype).reshape(b, heads, c // heads, h * w) for x in (q, k, v)]
        q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
        weights = torch.softmax((q @ k.transpose(-1, -2)) * temperature.to(dtype), dim=-1)
        return (weights @ v).reshape(b, c, h, w)


class ChannelAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=False)
        self.local = nn.Conv2d(3 * dim, 3 * dim, 3, padding=1, groups=3 * dim, bias=False)
        self.project = nn.Conv2d(dim, dim, 1, bias=False)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, x):
        q, k, v = self.local(self.qkv(x)).chunk(3, dim=1)
        result = channel_attention(q, k, v, self.heads, self.temperature)
        return self.project(result.to(dtype=self.project.weight.dtype))


class GatedFeedForward(nn.Module):
    def __init__(self, dim, expansion):
        super().__init__()
        hidden = max(1, int(dim * expansion))
        self.input = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.local = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                               groups=hidden * 2, bias=False)
        self.output = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x):
        a, b = self.local(self.input(x)).chunk(2, 1)
        return self.output(F.gelu(a) * b)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, expansion):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm2d(dim), LayerNorm2d(dim)
        self.attention = ChannelAttention(dim, heads)
        self.ffn = GatedFeedForward(dim, expansion)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class SharedEncoder(nn.Module):
    def __init__(self, dim, heads, depth, expansion, activation_checkpointing=False):
        super().__init__()
        self.embedding = nn.Conv2d(1, dim, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, expansion) for _ in range(depth)])
        self.output_norm = LayerNorm2d(dim)
        self.activation_checkpointing = activation_checkpointing

    def forward(self, image):
        x = self.embedding(image)
        for block in self.blocks:
            if (self.activation_checkpointing and self.training and torch.is_grad_enabled()
                    and (x.requires_grad or any(p.requires_grad for p in block.parameters()))):
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.output_norm(x)


class ImageDecoder(nn.Module):
    """Four candidate inputs; no source-image output shortcut."""
    def __init__(self, dim, heads, depth, expansion, activation_checkpointing=False):
        super().__init__()
        self.merge = nn.Conv2d(dim * 4, dim, 1, bias=False)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, expansion) for _ in range(depth)])
        self.output = nn.Sequential(nn.Conv2d(dim, max(1, dim // 2), 3, padding=1),
                                    nn.LeakyReLU(.1),
                                    nn.Conv2d(max(1, dim // 2), 1, 3, padding=1))
        self.activation_checkpointing = activation_checkpointing

    def forward(self, *branches):
        if len(branches) != 4:
            raise ValueError("Decoder requires common, IR-private, VI-private, joint")
        x = self.merge(torch.cat(branches, dim=1))
        for block in self.blocks:
            if (self.activation_checkpointing and self.training and torch.is_grad_enabled()
                    and (x.requires_grad or any(p.requires_grad for p in block.parameters()))):
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return torch.sigmoid(self.output(x).float())
