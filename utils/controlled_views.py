"""Identifiable factor interventions and known-target observation recovery.
No fixed quadrant or frequency identifies a relation. No MSRS test image is used.
"""
import math
import torch
import torch.nn.functional as F
from .augment import degrade_pair
from .targets import pool


def blob(x,y):
    cx,cy=(torch.rand(2,device=x.device)-.5)*1.2
    rx,ry=.35+torch.rand(2,device=x.device)*.8
    return (((x-cx)/rx).square()+((y-cy)/ry).square()<1).float()


def pattern(x,y):
    angle=torch.rand((),device=x.device)*math.pi
    phase=torch.rand((),device=x.device)*2*math.pi
    frequency=.03+torch.rand((),device=x.device)*.32
    coord=x*math.cos(float(angle))+y*math.sin(float(angle))
    pixels=max(x.shape[-1],y.shape[-2])/2
    if torch.rand((),device=x.device)<.5:
        return torch.sin(coord*2*math.pi*frequency*pixels+phase)
    return torch.tanh((coord-.3*torch.sin(phase))*12)


@torch.no_grad()
def make_controls(batch,height,width,device,config,natural_sources=None):
    y,x=torch.meshgrid(torch.linspace(-1,1,height,device=device),torch.linspace(-1,1,width,device=device),indexing="ij")
    cs,aa,bb,signs,targets,families=[],[],[],[],[],[]
    offset=int(torch.randint(4,(),device=device))
    for index in range(batch):
        family=(index+offset)%4
        c=.10*pattern(x,y)*blob(x,y)
        a=.07*pattern(x,y)*blob(x,y)
        b=.07*pattern(x,y)*blob(x,y)
        sign=-1. if torch.rand((),device=device)<.3 else 1.
        if family==0: a=torch.zeros_like(a);b=torch.zeros_like(b)  # Same content, possibly inverted contrast.
        if family==1:
            # Both roles can overlap the common and each other at the SAME pixel.
            a=.07*pattern(x,y);b=.07*pattern(x,y)
        if family==2:
            if natural_sources is not None:
                bank=natural_sources[int(torch.randint(2,(),device=device))]
                image=bank[int(torch.randint(len(bank),(),device=device)),0]
            else: image=.45+.12*pattern(x,y)+.06*pattern(x,y)
            split=float(torch.rand((),device=device)-.5)
            mi=(x<split+.25).float();mv=(x>split-.25).float()
            signal=image-.45
            c=signal*mi*mv;a=signal*mi*(1-mv);b=signal*mv*(1-mi)
            sign=1.
        cs.append(c);aa.append(a);bb.append(b);signs.append(sign)
        targets.append(.45+c+a+b);families.append(family)
    stack=lambda values:torch.stack(values)[:,None]
    c,a,b=stack(cs),stack(aa),stack(bb)
    sign=torch.tensor(signs,device=device)[:,None,None,None]
    ri,rv=.45+c+a,.45+sign*c+b
    target=stack(targets)
    if target.min()<0 or target.max()>1: raise ValueError("Generated target outside [0,1]")
    compatible=torch.ones_like(ri)
    correlated=torch.zeros_like(ri)
    source_reference_vi=rv.clone()
    shift_offsets=torch.zeros(batch,2,device=device)
    geometry_mask=torch.zeros_like(ri)
    for index,family in enumerate(families):
        if family!=3: continue
        # Geometry precedes photometric corruption. Keep aligned role references
        # separate from a source's own clean, displaced observation.
        offsets=(torch.rand(2,device=device)*2-1)*max(1.,min(height,width)/16)
        if offsets.abs().max()<.5: offsets[0]=.5
        shift_offsets[index]=offsets
        with torch.autocast(device_type=device.type,enabled=False):
            theta=torch.eye(2,3,device=device)[None]
            theta[0,0,2]=2*offsets[0]/width
            theta[0,1,2]=2*offsets[1]/height
            grid=F.affine_grid(theta,c[index:index+1].shape,align_corners=False)
            displaced=F.grid_sample(c[index:index+1],grid,padding_mode="border",align_corners=False)
        change=sign[index:index+1]*(displaced-c[index:index+1])
        source_reference_vi[index:index+1]=rv[index:index+1]+change
        mask=(pool(change.abs())>.005).float()
        compatible[index:index+1]=1-mask
        geometry_mask[index:index+1]=mask
    data=degrade_pair(ri,source_reference_vi,config,fusion_target=target,fusion_valid=torch.ones_like(ri))
    data["reference_vi"]=rv  # Aligned canonical pair for known-factor intervention ONLY.
    data["degradation_vi"]=torch.maximum(data["degradation_vi"],geometry_mask)
    for index,family in enumerate(families):
        if family==0 and torch.rand((),device=device)<.5:
            # Agreement on the same unwanted noise is not automatic reliability.
            scale=c[index:index+1].square().mean().sqrt().clamp_min(.03)
            interference=torch.randn_like(c[index:index+1])*scale
            for suffix in ("ir","vi"):
                data[suffix][index:index+1]=(data[suffix][index:index+1]+interference).clamp(0,1)
                data["degradation_"+suffix][index:index+1]=1
            correlated[index:index+1]=1
    data.update(factor_common=c,private_factor_ir=a,private_factor_vi=b,
        support_target=(pool(c.abs())>.005).float(),support_valid=torch.ones_like(ri),
        compatibility_target=compatible,compatibility_valid=torch.ones_like(ri),
        without_common=(ri-c,rv-sign*c),without_ir=(ri-a,rv),without_vi=(ri,rv-b),
        known=True,family=torch.tensor(families,device=device),correlated_interference=correlated,
        source_reference_ir=ri,source_reference_vi=source_reference_vi,shift_offsets=shift_offsets,
        source_absolute_valid=(torch.tensor(families,device=device)!=2)[:,None,None,None].float().expand_as(ri))
    return data
