"""Candidate fidelity and source-reference trust have distinct targets."""
import torch
from torch import nn
from .decomposition import head
from utils.operators import image_structure


class SourceTrust(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Conv2d(5,24,3,padding=1),nn.GELU(),
            nn.Conv2d(24,24,3,padding=2,dilation=2),nn.GELU(),nn.Conv2d(24,1,1))

    def forward(self,image,structure=None):
        x=image.detach()
        structure=image_structure(x) if structure is None else structure.detach()
        return torch.sigmoid(self.network(torch.cat((x,structure),1)).float())


class CandidateReliability(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.compress=nn.Conv2d(dim,16,1)
        self.common=head(3*16+9,1)
        self.private_ir,self.private_vi=head(3*16+10,1),head(3*16+10,1)
        self.source=SourceTrust()

    def forward(self,relation,ir,vi,include_source=True):
        c=self.compress(relation["shared"].detach())
        fi=self.compress(relation["feature_ir"].detach())
        fv=self.compress(relation["feature_vi"].detach())
        si,sv=relation["source_structure_ir"],relation["source_structure_vi"]
        support=relation["support"].detach()
        qc=torch.sigmoid(self.common(torch.cat((c,fi,fv,si,sv,support),1)).float())
        result={"quality_common":qc}
        if include_source:
            result.update(source_quality_ir=self.source(ir,si),source_quality_vi=self.source(vi,sv))
        for suffix,structure,feature,other,block in (
            ("ir",si,fi,sv,self.private_ir),("vi",sv,fv,si,self.private_vi)):
            p=self.compress(relation[f"private_{suffix}"].detach())
            cues=torch.cat((p,feature,c,structure,other,support,relation[f"novelty_{suffix}"].detach()),1)
            result[f"quality_private_{suffix}"]=torch.sigmoid(block(cues).float())
        return result
