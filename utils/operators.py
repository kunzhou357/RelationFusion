"""Fixed image operators, evaluated outside low precision autocast."""

import torch
from torch import nn
import torch.nn.functional as F


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("kernels", torch.stack((x, x.t()))[:, None], persistent=False)

    def forward(self, image):
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float() if image.dtype != torch.float64 else image
            channels = image.shape[1]
            kernels = self.kernels.to(image).repeat(channels, 1, 1, 1)
            return F.conv2d(image, kernels, padding=1, groups=channels).reshape(
                image.shape[0], channels, 2, *image.shape[-2:]
            )

    def magnitude(self, image):
        return self(image).abs().sum(dim=2)


def gaussian(image, sigma=1.0):
    """Separable Gaussian with replicated boundaries, including tiny images."""
    import math
    radius = max(1, math.ceil(3*sigma))
    with torch.autocast(device_type=image.device.type, enabled=False):
        image = image.float()
        axis = torch.arange(-radius, radius+1, device=image.device, dtype=image.dtype)
        weights = torch.exp(-axis.square()/(2*sigma*sigma))
        weights = weights/weights.sum()
        c = image.shape[1]
        x = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"),
                     weights[None, None, None, :].expand(c, 1, 1, -1), groups=c)
        return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"),
                        weights[None, None, :, None].expand(c, 1, -1, 1), groups=c)


def signed_gradient(image):
    with torch.autocast(device_type=image.device.type, enabled=False):
        image = image.float()
        kernel = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/8
        kernels = torch.stack((kernel, kernel.t()))[:, None].repeat(image.shape[1], 1, 1, 1)
        result = F.conv2d(F.pad(image, (1, 1, 1, 1), mode="replicate"), kernels, groups=image.shape[1])
        return result.reshape(image.shape[0], image.shape[1], 2, *image.shape[-2:])


def structure_tensor(image):
    grad = signed_gradient(gaussian(image, .8))[:, 0]
    gx, gy = grad[:, :1], grad[:, 1:]
    average = lambda v: F.avg_pool2d(F.pad(v, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
    xx, yy, xy = average(gx.square()), average(gy.square()), average(gx*gy)
    coherence = ((xx-yy).square()+4*xy.square()+1e-12).sqrt()/(xx+yy+1e-6)
    coherence = coherence.clamp(0, 1)
    return grad, coherence, (xx+yy+1e-12).sqrt()


def image_structure(image):
    grad, coherence, _ = structure_tensor(image)
    return torch.cat((gaussian(image, 2), grad, coherence), 1)


def ssim_map(x, y):
    """Spatial SSIM shared by source loss, fusion loss and gain labels."""
    x, y = x.float(), y.float()
    mx, my = gaussian(x, 1.5), gaussian(y, 1.5)
    vx, vy = gaussian(x*x, 1.5)-mx*mx, gaussian(y*y, 1.5)-my*my
    cov = gaussian(x*y, 1.5)-mx*my
    return (((2*mx*my+.01**2)*(2*cov+.03**2))/
            ((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).clamp(-1, 1)
