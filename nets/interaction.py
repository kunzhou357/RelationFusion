"""Multiplicative cross interaction with exact zero-source contracts."""
import torch
from torch import nn
import torch.nn.functional as F
from .blocks import channel_attention


def normalize(x):
    return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)


class CrossChannelAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.query = nn.Conv2d(dim, dim, 1, bias=False)
        self.key_value = nn.Conv2d(dim, 2*dim, 1, bias=False)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.output = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, source, context):
        q = self.query(normalize(source))
        k, v = self.key_value(normalize(context)).chunk(2, 1)
        return self.output(channel_attention(q, k, v, self.heads, self.temperature))


class JointInteraction(nn.Module):
    def __init__(self, dim, heads, mode="cross"):
        super().__init__()
        self.mode = mode
        if mode != "none":
            self.input_ir = nn.Conv2d(dim, dim, 1, bias=False)
            self.input_vi = nn.Conv2d(dim, dim, 1, bias=False)
        if mode == "cross":
            self.phi_ir = nn.Conv2d(dim, dim, 1, bias=False)
            self.phi_vi = nn.Conv2d(dim, dim, 1, bias=False)
            self.ir_query, self.vi_query = CrossChannelAttention(dim, heads), CrossChannelAttention(dim, heads)
            self.merge = nn.Conv2d(3*dim, dim, 1, bias=False)
        elif mode == "conv":
            hidden = max(1, round((13*dim*dim+2*heads)/(19*dim)))
            self.conv = nn.Sequential(nn.Conv2d(2*dim, hidden, 3, padding=1, bias=False),
                                      nn.GELU(), nn.Conv2d(hidden, dim, 1, bias=False))

    def from_inputs(self, x_ir, x_vi):
        if self.mode == "none":
            return torch.zeros_like(x_ir)
        if self.mode == "conv":
            return self.conv(torch.cat((x_ir, x_vi), 1))
        a, b = self.phi_ir(x_ir), self.phi_vi(x_vi)
        return self.merge(torch.cat((a*b, a*self.ir_query(x_ir, x_vi), b*self.vi_query(x_vi, x_ir)), 1))

    def forward(self, relation):
        if self.mode == "none":
            return torch.zeros_like(relation["shared"])
        inputs = []
        for suffix, project in (("ir", self.input_ir), ("vi", self.input_vi)):
            inputs.append(project(relation[f"unexplained_{suffix}"]))
        return self.from_inputs(*inputs)
