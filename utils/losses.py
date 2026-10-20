"""Task losses with explicit ownership of content, membership and scoring gradients."""
import torch
from torch import nn
import torch.nn.functional as F
from .targets import (charbonnier,pool,differences,masked_mean,fusion_loss,
                      fidelity_target,action_target,fidelity_settings)
from .operators import ssim_map


def balanced_mean(value,target,valid,bins=4):
    # Equally weight occupied quality bins, not the huge unchanged background.
    terms=[]
    for index in range(bins):
        mask=valid*((target>=index/bins)&(target<((index+1)/bins) if index<bins-1 else target<=1)).float()
        if bool(mask.any()): terms.append(masked_mean(value,mask))
    return torch.stack(terms).mean() if terms else value.sum()*0


def probability_loss(prediction,target,valid):
    with torch.autocast(device_type=prediction.device.type,enabled=False):
        value=F.binary_cross_entropy(prediction.float(),target.float(),reduction="none")
    return balanced_mean(value,target,valid,2)


def factor_error(prediction,target):
    valid=(pool(target.abs())>.005).float()
    loss=masked_mean(charbonnier(prediction-target),valid)
    loss=loss+.25*masked_mean(charbonnier(prediction-target),1-valid)
    loss=loss+.25*sum(charbonnier(a-b).mean() for a,b in zip(differences(prediction),differences(target)))
    return loss


class StructuralSimilarity(nn.Module):
    def forward(self,x,y): return ssim_map(x,y).mean()


class FusionObjectives(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config

    def source(self,output,ir,vi):
        return sum(charbonnier(output["reconstruction_"+s]-x).mean()
            +.1*(1-ssim_map(output["reconstruction_"+s],x)).mean()
            for s,x in (("ir",ir),("vi",vi)))/2

    def fusion(self,output,data):
        data=self.task(data)
        if "fused_off" in output:
            return .5*(fusion_loss(output["fused_off"],data)+fusion_loss(output["fused_on"],data))
        return fusion_loss(output["fused"],data)

    def task(self,data):
        return dict(data,known_gradient_weight=self.config.known_gradient_weight,
                    known_ssim_weight=self.config.known_ssim_weight)

    def roles(self,model,data,intervention_key="without_common"):
        # Live memberships here; the main fusion graph detaches them.
        clean=model.factorize(data["reference_ir"],data["reference_vi"],False)
        truth={"common":data["factor_common"],"ir":data["private_factor_ir"],"vi":data["private_factor_vi"]}
        content=sum(factor_error(clean["readout_"+s],target) for s,target in truth.items())/3
        support=probability_loss(clean["support"],data["support_target"],data["support_valid"])
        alt=model.factorize(*data[intervention_key],detach_membership=False)
        if intervention_key=="without_common":
            targets={"common":torch.zeros_like(truth["common"]),"ir":truth["ir"],"vi":truth["vi"]}
            invariant=("private_ir","private_vi")
        else:
            suffix="ir" if intervention_key=="without_ir" else "vi"
            targets=dict(truth);targets[suffix]=torch.zeros_like(truth[suffix])
            invariant=("shared","private_vi" if suffix=="ir" else "private_ir")
        content=(content+sum(factor_error(alt["readout_"+s],t) for s,t in targets.items())/3)/2
        invariant_loss=content*0
        for k in invariant:
            scale=(clean[k].detach()-clean[k].detach().mean((-2,-1),keepdim=True)).square().mean().sqrt().clamp_min(.05)
            invariant_loss=invariant_loss+F.smooth_l1_loss(alt[k]/scale,clean[k]/scale)
        return {"role":content,"intervention":invariant_loss/len(invariant),"support":support}

    @torch.no_grad()
    def candidate_targets(self,output,data):
        return {s:fidelity_target(output["readout_"+s],target,**fidelity_settings(self.config))
            for s,target in (("common",data["factor_common"]),("ir",data["private_factor_ir"]),("vi",data["private_factor_vi"]))}

    def candidate_quality(self,output,targets):
        terms=[]
        for suffix,key in (("common","quality_common"),("ir","quality_private_ir"),("vi","quality_private_vi")):
            target,valid=targets[suffix]
            terms.append(balanced_mean((output[key]-target).abs(),target,valid))
        return torch.stack(terms).mean()

    def scores(self,model,output,data,tamper=True):
        targets=self.candidate_targets(output,data)
        zero=output["shared"].new_zeros(())
        quality=zero
        if model.config.verification:
            quality=self.candidate_quality(output,targets)
            source_loss=zero
            for s in ("ir","vi"):
                source_reference=data.get("source_reference_"+s,data["reference_"+s])
                target,valid=fidelity_target(data[s],source_reference,source=True,**fidelity_settings(self.config))
                valid=valid*data["source_absolute_valid"]
                source_loss=source_loss+balanced_mean((output["source_quality_"+s]-target).abs(),target,valid)
            quality=quality+.5*source_loss
            if tamper:
                changed=dict(output)
                key=("shared","private_ir","private_vi")[int(torch.randint(3,()))]
                value=changed[key].detach()
                ac=value-value.mean((-2,-1),keepdim=True)
                scale=ac.square().mean().sqrt().clamp_min(.03)
                changed[key]=torch.roll(value,max(1,value.shape[-1]//3),-1)+torch.randn_like(value)*scale
                with torch.no_grad(): changed.update(model.decomposition.readouts(changed,detach=True))
                changed.update(model.verification(changed,data["ir"],data["vi"],include_source=False))
                quality=(quality+.5*self.candidate_quality(changed,self.candidate_targets(changed,data)))/1.5
        evidence=zero
        if model.config.decision:
            for s in ("ir","vi"):
                presence=(pool(data["private_factor_"+s].abs())>.01).float()
                desired=model.config.evidence_scale*presence*targets[s][0]
                evidence=evidence+F.smooth_l1_loss(torch.log1p(output["evidence_"+s]),torch.log1p(desired))
            evidence=evidence/2
        compatibility=probability_loss(output["compatibility"],data["compatibility_target"],data["compatibility_valid"]) if model.config.compatibility else zero
        return {"quality":quality,"evidence":evidence,"compatibility":compatibility}

    def natural_source_ranking(self,model,data,output=None):
        # Relative added-distortion constraint only; no natural reference Q=1.
        if not model.config.verification: return data["ir"].sum()*0
        result=0
        for s in ("ir","vi"):
            altered=output["source_quality_"+s] if output is not None else model.verification.source(data[s])
            original=model.verification.source(data["reference_"+s])
            fidelity,_=fidelity_target(data[s],data["reference_"+s],source=True,**fidelity_settings(self.config))
            mask=data["degradation_"+s]
            margin=.1*(1-fidelity)
            result=result+masked_mean(F.relu(altered-original+margin),mask)
        return result/2

    def gain(self,output,data):
        target,valid=action_target(output,self.task(data),self.config.gain_tolerance)
        value=-(target*output["gain_logits"].log_softmax(1)).sum(1,keepdim=True)
        return masked_mean(value,valid)
