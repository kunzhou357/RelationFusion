"""Source-owned values, bilateral channel membership and an explicit remainder.
Fusion detaches membership. Known-factor graphs do NOT detach membership.
"""
import torch
from torch import nn
import torch.nn.functional as F
from utils.operators import image_structure


def head(inputs,outputs,hidden=32):
    return nn.Sequential(nn.Conv2d(inputs,hidden,3,padding=1),nn.GELU(),nn.Conv2d(hidden,outputs,1))


def content_block(inputs,outputs):
    return nn.Sequential(nn.Conv2d(inputs,outputs,1,bias=False),
        nn.Conv2d(outputs,outputs,3,padding=1,groups=outputs,bias=False),nn.GELU(),
        nn.Conv2d(outputs,outputs,1,bias=False))


class FactorReadout(nn.Module):
    """Bias-free L1-bounded projection: cannot amplify a collapsed candidate.

    The sum of absolute effective weights is <=1. A tiny content tensor cannot
    appear to contain a large image factor just by inflating an auxiliary head.
    """
    def __init__(self,dim):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(1,dim,1,1))
        nn.init.normal_(self.weight,std=.02)

    def forward(self,value):
        weights=self.weight.float()
        weights=weights/weights.abs().sum().clamp_min(1.)
        return F.conv2d(value,weights).float()


class RelationFactorization(nn.Module):
    def __init__(self,dim,common_dim):
        super().__init__()
        self.source_ir=content_block(dim,common_dim)
        self.source_vi=content_block(dim,common_dim)
        self.common_membership=head(3*common_dim+8,2*common_dim)
        self.support=head(3*common_dim+8,1)
        self.common_project=nn.Conv2d(common_dim,dim,1,bias=False)
        self.private_content_ir=content_block(dim,dim)
        self.private_content_vi=content_block(dim,dim)
        self.private_membership_ir=head(2*dim+8,dim)
        self.private_membership_vi=head(2*dim+8,dim)
        # Small probes have no image/other-branch shortcut.
        self.readout_common=FactorReadout(dim)
        self.readout_ir=FactorReadout(dim)
        self.readout_vi=FactorReadout(dim)

    def readouts(self,relation,detach=False):
        get=lambda k: relation[k].detach() if detach else relation[k]
        return {"readout_common":self.readout_common(get("shared")).float(),
                "readout_ir":self.readout_ir(get("private_ir")).float(),
                "readout_vi":self.readout_vi(get("private_vi")).float()}

    def forward(self,fi,fv,ir,vi,detach_membership=True,membership_blend=1.):
        hi,hv=self.source_ir(fi),self.source_vi(fv)
        si,sv=image_structure(ir.detach()),image_structure(vi.detach())
        cues=torch.cat((hi.detach(),hv.detach(),(hi-hv).detach().abs(),si,sv),1)
        mi,mv=torch.sigmoid(self.common_membership(cues).float()).chunk(2,1)
        support=torch.sigmoid(self.support(cues).float())
        use=lambda m:(1-membership_blend)+membership_blend*(m.detach() if detach_membership else m)
        common=self.common_project((use(mi)*hi+use(mv)*hv)*.5)
        result={"feature_ir":fi,"feature_vi":fv,"shared":common,"support":support,
                "common_view_ir":hi,"common_view_vi":hv,"common_membership_ir":mi,"common_membership_vi":mv,
                "source_structure_ir":si,"source_structure_vi":sv}
        for suffix,f,other,source,context,values,member in (
            ("ir",fi,fv,si,sv,self.private_content_ir,self.private_membership_ir),
            ("vi",fv,fi,sv,si,self.private_content_vi,self.private_membership_vi)):
            m=torch.sigmoid(member(torch.cat((f.detach(),other.detach(),source,context),1)).float())
            private=use(m)*values(f)
            # Fixed identity coordinate maps avoid unconstrained learned O adapters.
            residual=f-common
            result.update({f"common_mapped_{suffix}":common,f"residual_{suffix}":residual,
                f"private_membership_{suffix}":m,f"novelty_{suffix}":m.mean(1,keepdim=True),
                f"private_{suffix}":private,f"unexplained_{suffix}":residual-private})
        result.update(self.readouts(result))
        return result
