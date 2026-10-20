"""Availability evidence, compatibility and optional injection-action policy."""
import torch
from torch import nn
import torch.nn.functional as F
from .decomposition import head
from .spatial_attention import MDA
from .spectral_gate import SpectralDescriptors
from .beta_decision import BetaEvidenceDecision


class RelationDecision(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.spectral=SpectralDescriptors(config.patch_size)
        self.spatial=MDA(config.dim,max_tokens=config.attention_tokens)
        self.evidence_ir,self.evidence_vi=head(9,1),head(9,1)
        self.compatibility=head(10,1)
        self.beta=BetaEvidenceDecision(config.lambda_prior)
        if config.learned_gain:
            self.gain_compress=nn.Conv2d(config.dim,8,1)
            self.gain=head(6*8+8+6,3)

    def forward(self,relation,ir,vi):
        pir,pvi=relation["private_ir"].detach(),relation["private_vi"].detach()
        normalized,spectra=self.spectral(pir,pvi)
        attention=self.spatial(*normalized)
        structures=(relation["source_structure_ir"],relation["source_structure_vi"])
        evidence=[]
        for i,block in enumerate((self.evidence_ir,self.evidence_vi)):
            evidence.append(F.softplus(block(torch.cat((spectra[i],attention,structures[i]),1)).float()))
        stats=self.beta(*evidence)
        # Exactly one uncertainty calibration from the RAW Beta mean.
        weight=self.beta.calibrate_spatial(stats["weight_raw"],stats["uncertainty"])
        if not self.config.decision: weight=torch.full_like(weight,.5)
        k=torch.sigmoid(self.compatibility(torch.cat((*structures,spectra[0][:,3:4],relation["support"].detach()),1)).float())
        if not self.config.compatibility: k=torch.ones_like(k)
        kd=k.detach()
        return {"evidence_ir":evidence[0],"evidence_vi":evidence[1],"weight_ir":1-weight,"weight_vi":weight,
            "uncertainty":stats["uncertainty"],"compatibility":k,
            "contribution_ir":kd+(1-kd)*(1-weight),"contribution_vi":kd+(1-kd)*weight}

    def joint_gain(self,relation,joint,ir,vi):
        if not self.config.learned_gain: raise ValueError("Model has no injection policy")
        tensors=[relation[k] for k in ("shared","private_ir","private_vi","unexplained_ir","unexplained_vi")]+[joint]
        compressed=[self.gain_compress(x.detach()) for x in tensors]
        base_decision=[relation[k].detach() for k in
            ("quality_common","quality_private_ir","quality_private_vi","compatibility","weight_vi","uncertainty")]
        logits=self.gain(torch.cat((*compressed,relation["source_structure_ir"],relation["source_structure_vi"],*base_decision),1)).float()
        p=logits.softmax(1)
        strength=(p*p.new_tensor([0.,.5,1.])[None,:,None,None]).sum(1,keepdim=True)
        return logits,strength
