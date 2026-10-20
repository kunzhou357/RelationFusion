"""Training-only degradation. References never become network inputs by accident."""
import torch
from .operators import gaussian


@torch.no_grad()
def degrade_pair(ir, vi, config, fusion_target=None, fusion_valid=None):
    refs = [ir.detach().clone(), vi.detach().clone()]
    inputs = [x.clone() for x in refs]
    masks = [torch.zeros_like(ir), torch.zeros_like(vi)]
    types = torch.full((ir.shape[0], 2), -1, device=ir.device, dtype=torch.long)
    h, w = ir.shape[-2:]
    for index in range(ir.shape[0]):
        if torch.rand((), device=ir.device).item() >= config.degrade_probability:
            continue
        mode = int(torch.randint(3, (), device=ir.device))
        for modality in ((0,) if mode == 0 else (1,) if mode == 1 else (0, 1)):
            kind = int(torch.randint(4, (), device=ir.device))
            size_y = int(torch.randint(max(1, h//4), h+1, (), device=ir.device))
            size_x = int(torch.randint(max(1, w//4), w+1, (), device=ir.device))
            y = int(torch.randint(h-size_y+1, (), device=ir.device))
            x = int(torch.randint(w-size_x+1, (), device=ir.device))
            mask = masks[modality][index:index+1]
            mask[..., y:y+size_y, x:x+size_x] = 1
            source = refs[modality][index:index+1]
            severity = .2+.8*torch.rand((), device=ir.device)
            if kind == 0:
                altered = source+torch.randn_like(source)*config.noise_max*severity
            elif kind == 1:
                altered = gaussian(source, .3+config.blur_max*float(severity))
            elif kind == 2:
                axis = torch.arange(w, device=ir.device)[None, None, None, :]
                frequency = 2+int(torch.randint(2, max(3, w//2+1), (), device=ir.device))
                phase = torch.rand((), device=ir.device)*6.283185
                altered = source+config.stripe_max*severity*torch.sin(axis*6.283185*frequency/max(w, 1)+phase)
            else:
                scale = 1-severity*(1-config.contrast_min)
                altered = source.mean((-2, -1), keepdim=True)+scale*(source-source.mean((-2, -1), keepdim=True))
            inputs[modality][index:index+1] = torch.where(mask.bool(), altered.clamp(0, 1), source)
            types[index, modality] = kind
    return {"ir": inputs[0], "vi": inputs[1], "reference_ir": refs[0], "reference_vi": refs[1],
            "degradation_ir": masks[0], "degradation_vi": masks[1], "degradation_type": types,
            "fusion_target": fusion_target,
            "fusion_valid": torch.ones_like(ir) if fusion_valid is None else fusion_valid}
