"""Scale-normalized patch spectra. Descriptors are not quality probabilities."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def split_patches(value, size):
    b, c, h, w = value.shape
    dh, dw = (-h) % size, (-w) % size
    value = F.pad(value.float(), (0, dw, 0, dh), mode="replicate")
    nh, nw = value.shape[-2]//size, value.shape[-1]//size
    patches = value.reshape(b, c, nh, size, nw, size).permute(0, 2, 4, 1, 3, 5)
    return patches, (h, w)


def join_patches(value, shape):
    b, nh, nw, c, ph, pw = value.shape
    result = value.permute(0, 3, 1, 4, 2, 5).reshape(b, c, nh*ph, nw*pw)
    return result[..., :shape[0], :shape[1]]


class SpectralDescriptors(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, a, b):
        with torch.autocast(device_type=a.device.type, enabled=False):
            patches, shape = zip(*(split_patches(x, self.patch_size) for x in (a, b)))
            normalized = []
            for x in patches:
                ac = x-x.mean((-2, -1), keepdim=True)
                normalized.append(ac/ac.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(.1))
            spectra = [torch.fft.rfft2(x, norm="ortho") for x in normalized]
            powers = [s.abs().square().mean(3) for s in spectra]
            phase = (spectra[0]*spectra[1].conj()).real
            phase = phase/(spectra[0].abs()*spectra[1].abs()+1e-6)
            phase = phase.mean((3, 4, 5))
            descriptors = []
            for index, power in enumerate(powers):
                probability = power/(power.sum((-2, -1), keepdim=True)+1e-8)
                entropy = -(probability*probability.clamp_min(1e-8).log()).sum((-2, -1))
                entropy = entropy/max(math.log(power.shape[-1]*power.shape[-2]), 1)
                energy = power.mean((-2, -1))
                difference = (power-powers[1-index]).relu().mean((-2, -1))
                stats = torch.stack((entropy, energy/(1+energy), difference/(1+difference), phase), -1)
                bsz, nh, nw, _ = stats.shape
                stats = stats.permute(0, 3, 1, 2).repeat_interleave(self.patch_size, -2).repeat_interleave(self.patch_size, -1)
                descriptors.append(stats[..., :shape[index][0], :shape[index][1]])
            return [join_patches(x, shape[i]) for i, x in enumerate(normalized)], descriptors
