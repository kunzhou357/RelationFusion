"""Fixed image-domain supervision; learned candidate gates cannot rewrite targets."""
import torch
import torch.nn.functional as F
from .operators import gaussian,ssim_map


def charbonnier(value): return (value.float().square()+1e-6).sqrt()-.001


def pool(value,size=5):
    r=size//2
    return F.avg_pool2d(F.pad(value,(r,r,r,r),mode="replicate"),size,stride=1)


def differences(value):
    x=value.float()
    dx=F.pad(x.diff(dim=-1),(0,1,0,0)) if x.shape[-1]>1 else torch.zeros_like(x)
    dy=F.pad(x.diff(dim=-2),(0,0,0,1)) if x.shape[-2]>1 else torch.zeros_like(x)
    return dx,dy


def masked_mean(value,mask):
    mask=mask.detach().to(value).expand_as(value)
    return (value*mask).sum()/mask.sum().clamp_min(1)


@torch.no_grad()
def natural_reference(ir,vi,trust=None,threshold=.7,sigma=4.,low_weight=.25,detail_floor=.001):
    ones=torch.ones_like(ir)
    masks=[ones,ones] if trust is None else [(q>=threshold).float() for q in trust]
    grads=[differences(x) for x in (ir,vi)]
    result={}
    for axis,label in ((-1,"x"),(-2,"y")):
        edges=[]
        for m in masks:
            neighbor=torch.roll(m,-1,dims=axis)
            if axis==-1: neighbor[...,-1]=m[...,-1]
            else: neighbor[...,-1,:]=m[...,-1,:]
            edges.append(torch.minimum(m,neighbor))
        index=0 if label=="x" else 1
        result["detail_"+label]=torch.maximum(edges[0]*grads[0][index].abs(),edges[1]*grads[1][index].abs())
        supported=torch.maximum(edges[0]*(grads[0][index].abs()>detail_floor),
                                edges[1]*(grads[1][index].abs()>detail_floor))
        # A trusted flat source does not prove another source's unique detail is
        # absent. Both-trusted flat observations may constrain detail to zero;
        # a single trusted source must actually supply that directional detail.
        result["exact_"+label]=edges[0]*edges[1]
        result["valid_"+label]=torch.maximum(result["exact_"+label],supported)
        # Forward differences have no pair at the last column/row.
        if label=="x":
            result["valid_x"][...,-1]=0
            result["exact_x"][...,-1]=0
        else:
            result["valid_y"][...,-1,:]=0
            result["exact_y"][...,-1,:]=0
    lm=[(gaussian(m,sigma)>.99).float() for m in masks]
    result["low"]=torch.maximum(lm[0]*gaussian(ir,sigma),lm[1]*gaussian(vi,sigma))
    # The max-brightness prior is unspecified when a source is rejected.
    result["low_valid"]=lm[0]*lm[1]
    result.update(low_sigma=sigma,low_weight=low_weight)
    return result


def detail_penalty(gradient,target,exact):
    magnitude=gradient.abs()
    # A single trusted source supplies positive evidence, not a complete upper
    # bound on the other source's unique detail. Both-trusted retains the max prior.
    error=torch.where(exact.bool(),magnitude-target,F.relu(target-magnitude))
    return charbonnier(error)


def fusion_error_map(fused,data):
    if data.get("fusion_target") is not None:
        target=data["fusion_target"].detach()
        valid=data["fusion_valid"].detach()
        # Invalid target pixels must not contaminate adjacent gradient/SSIM terms.
        estimate=fused*valid+target*(1-valid)
        gradient=sum(charbonnier(a-b) for a,b in zip(differences(estimate),differences(target)))
        return (charbonnier(estimate-target)+data.get("known_gradient_weight",1.)*gradient
                +data.get("known_ssim_weight",.1)*(1-ssim_map(estimate,target)))
    proxy=data["proxy"]
    low=proxy["low_weight"]*charbonnier(gaussian(fused,proxy["low_sigma"])-proxy["low"])*proxy["low_valid"]
    dx,dy=differences(fused)
    return (low+detail_penalty(dx,proxy["detail_x"],proxy["exact_x"])*proxy["valid_x"]
            +detail_penalty(dy,proxy["detail_y"],proxy["exact_y"])*proxy["valid_y"])


def fusion_loss(fused,data):
    if data.get("fusion_target") is not None:
        return masked_mean(fusion_error_map(fused,data),data["fusion_valid"])
    p=data["proxy"]
    dx,dy=differences(fused)
    # Normalize each valid term separately. No penalty towards zero if rejected.
    return (p["low_weight"]*masked_mean(charbonnier(gaussian(fused,p["low_sigma"])-p["low"]),p["low_valid"])
        +masked_mean(detail_penalty(dx,p["detail_x"],p["exact_x"]),p["valid_x"])
        +masked_mean(detail_penalty(dy,p["detail_y"],p["exact_y"]),p["valid_y"]))


@torch.no_grad()
def fidelity_target(estimate,target,tau=1.,source=False,intensity_floor=.05,edge_floor=.01,source_floor=.03,edge_weight=.5):
    estimate,target=estimate.detach().float(),target.detach().float()
    denominator=pool((target-pool(target)).abs())+source_floor if source else pool(target.abs())+intensity_floor
    error=pool((estimate-target).abs())/denominator
    for a,b in zip(differences(estimate),differences(target)):
        error=error+edge_weight*pool((a-b).abs())/(pool(b.abs())+edge_floor)
    quality=torch.exp(-error/tau)
    valid=torch.ones_like(quality) if source else ((pool(target.abs())>.005)|(pool(estimate.abs())>.005)).float()
    return quality,valid


def fidelity_settings(config):
    return dict(tau=config.quality_tau,intensity_floor=config.quality_intensity_floor,
                edge_floor=config.quality_edge_floor,source_floor=config.source_contrast_floor,
                edge_weight=config.quality_edge_weight)


@torch.no_grad()
def action_target(output,data,tolerance=.001):
    if data.get("fusion_target") is None:
        raise ValueError("Injection actions require a known target, not the natural weak prior")
    errors=torch.cat([pool(fusion_error_map(output[key],data)) for key in ("fused_off","fused_half","fused_on")],1)
    minimum=errors.min(1,keepdim=True).values
    probabilities=torch.softmax(-(errors-minimum)/max(tolerance,1e-6),1)
    negligible=(errors[:,:1]-minimum)<=tolerance
    off=torch.zeros_like(probabilities);off[:,0]=1
    return torch.where(negligible,off,probabilities),data["fusion_valid"].detach()
