"""Four relation modules with persistent content learning and optional calibration."""
import torch
from torch import nn
from .blocks import SharedEncoder, ImageDecoder
from .decomposition import RelationFactorization
from .verification import CandidateReliability, SourceTrust
from .interaction import JointInteraction
from .decision import RelationDecision
from utils.settings import ModelConfig, STAGES

MODEL_VERSION="relation_v3"


class ReconstructionAdapter(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.modality=nn.Parameter(torch.randn(2,dim,1,1)*.02)
        self.mapping=nn.Conv2d(2*dim,4*dim,1)

    def forward(self,feature,modality):
        token=self.modality[modality].unsqueeze(0).expand_as(feature)
        return self.mapping(torch.cat((feature,token),1)).chunk(4,1)


class RelationFusion(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        self.config=c=config or ModelConfig()
        self.encoder=SharedEncoder(c.dim,c.heads,c.encoder_blocks,c.expansion,c.activation_checkpointing)
        self.decoder=ImageDecoder(c.dim,c.heads,c.decoder_blocks,c.expansion,c.activation_checkpointing)
        self.reconstruction=ReconstructionAdapter(c.dim)
        self.decomposition=RelationFactorization(c.dim,c.common_dim)
        self.verification=CandidateReliability(c.dim)
        self.interaction=JointInteraction(c.dim,c.heads,c.interaction)
        self.decision=RelationDecision(c)
        self.structure=nn.Conv2d(c.dim,c.dim,1,bias=False)
        self.private_ir=nn.Conv2d(c.dim,c.dim,1,bias=False)
        self.private_vi=nn.Conv2d(c.dim,c.dim,1,bias=False)
        self.joint_map=nn.Conv2d(c.dim,c.dim,1,bias=False)
        nn.init.normal_(self.joint_map.weight,std=.001)
        self.reference_teacher=SourceTrust()
        self.reference_teacher.requires_grad_(False)
        self.register_buffer("reference_ready",torch.tensor(False))
        self.register_buffer("reference_attempted",torch.tensor(False))
        self.register_buffer("gain_updates",torch.zeros((),dtype=torch.long))
        self.register_buffer("joint_updates",torch.zeros((),dtype=torch.long))
        self.register_buffer("fusion_updates",torch.zeros((),dtype=torch.long))
        self.register_buffer("gate_strength",torch.tensor(1.))
        self.register_buffer("joint_strength",torch.tensor(1.))
        self.stage=None

    @staticmethod
    def _validate(ir,vi):
        if ir.ndim!=4 or ir.shape!=vi.shape or ir.shape[1]!=1 or min(ir.shape[-2:])<1:
            raise ValueError("Inputs need identical nonempty [B,1,H,W] shapes")
        if ir.device!=vi.device or ir.dtype!=vi.dtype or not ir.is_floating_point():
            raise ValueError("Inputs must share floating dtype and device")

    @staticmethod
    def parameter_group(name):
        if name.startswith("reference_teacher."): return None
        if name.startswith("encoder."): return "encoder"
        if name.startswith(("decision.gain.","decision.gain_compress.")): return "gain"
        if name.startswith(("interaction.","joint_map.")): return "joint"
        if name.startswith("verification."): return "quality"
        if name.startswith("decision."): return "decision"
        if name.startswith(("decomposition.common_membership.","decomposition.private_membership_","decomposition.support.")):
            return "relation"
        return "content"

    def configure_stage(self,stage,gate_blend=1.,joint_blend=1.):
        if stage not in STAGES: raise ValueError("Unknown training stage")
        if stage=="calibration" and not self.config.learned_gain:
            raise ValueError("Calibration requires a configured injection policy")
        self.stage=stage
        self.gate_strength.fill_(gate_blend)
        self.joint_strength.fill_(joint_blend if stage in ("joint","calibration") else 0.)
        for name,p in self.named_parameters():
            group=self.parameter_group(name)
            if stage=="warmup": enabled=name.startswith(("encoder.","decoder.","reconstruction."))
            elif stage=="calibration": enabled=group=="gain"
            else: enabled=group is not None and group!="gain" and (group!="joint" or stage=="joint")
            if not self.config.verification and group=="quality": enabled=False
            if not self.config.decision and name.startswith(("decision.evidence_","decision.spatial.")): enabled=False
            if not self.config.compatibility and name.startswith("decision.compatibility."): enabled=False
            if self.config.interaction=="none" and group=="joint": enabled=False
            p.requires_grad_(enabled)
            p.grad=None
        self.train()

    def train(self,mode=True):
        super().train(mode)
        self.reference_teacher.eval()
        return self

    def configure_epoch(self,epoch,config):
        self.configure_stage(config.stage(epoch),config.gate_blend(epoch),config.joint_blend(epoch))

    @torch.no_grad()
    def freeze_reference(self):
        self.reference_teacher.load_state_dict(self.verification.source.state_dict())
        self.reference_teacher.requires_grad_(False)
        self.reference_ready.fill_(True)
        self.reference_attempted.fill_(True)

    @torch.no_grad()
    def reference_trust(self,ir,vi):
        if not bool(self.reference_ready): return None
        return self.reference_teacher(ir),self.reference_teacher(vi)

    def reconstruct(self,feature,modality):
        return self.decoder(*self.reconstruction(feature,modality))

    def warmup(self,ir,vi):
        self._validate(ir,vi)
        fi,fv=self.encoder(ir),self.encoder(vi)
        return {"feature_ir":fi,"feature_vi":fv,
            "reconstruction_ir":self.reconstruct(fi,0),"reconstruction_vi":self.reconstruct(fv,1)}

    def factorize(self,ir,vi,detach_membership=True,membership_blend=1.):
        self._validate(ir,vi)
        return self.decomposition(self.encoder(ir),self.encoder(vi),ir,vi,detach_membership,membership_blend)

    def assemble(self,relation,joint):
        blend=self.gate_strength
        qc=(1-blend)+blend*relation["quality_common"].detach()
        branches=[self.structure(qc*relation["shared"])]
        for suffix,project in (("ir",self.private_ir),("vi",self.private_vi)):
            q=(1-blend)+blend*relation[f"quality_private_{suffix}"].detach()
            a=(1-blend)+blend*relation[f"contribution_{suffix}"].detach()
            branches.append(project(a*q*relation[f"private_{suffix}"]))
        branches.append(self.joint_strength*self.joint_map(joint))
        return tuple(branches)

    def forward(self,ir,vi,return_aux=False,reconstruct=False,joint_mode="auto"):
        if joint_mode not in ("auto","off","on","probe","dual","gated"): raise ValueError("Invalid joint mode")
        output=self.factorize(ir,vi,True,float(self.gate_strength))
        if self.config.verification:
            output.update(self.verification(output,ir,vi,include_source=self.training or return_aux))
        else:
            output.update({k:torch.ones_like(ir) for k in
                ("quality_common","quality_private_ir","quality_private_vi","source_quality_ir","source_quality_vi")})
        output.update(self.decision(output,ir,vi))
        if joint_mode=="auto":
            joint_mode="gated" if self.gain_updates.item()>0 else "on" if self.joint_updates.item()>0 else "off"
        if joint_mode=="gated" and self.gain_updates.item()==0: raise ValueError("Injection policy is untrained")
        active=joint_mode!="off" and self.config.interaction!="none"
        joint=self.interaction(output) if active else torch.zeros_like(output["shared"])
        output["joint"]=joint
        branches=self.assemble(output,joint)
        gate=torch.ones_like(ir) if active else torch.zeros_like(ir)
        if self.config.learned_gain and active and (joint_mode in ("probe","gated")):
            logits,predicted=self.decision.joint_gain(output,joint,ir,vi)
            output["gain_logits"],output["predicted_gain"]=logits,predicted
            if joint_mode=="gated": gate=predicted.detach()
        if joint_mode in ("probe","dual"):
            off=self.decoder(*branches[:3],torch.zeros_like(branches[3]))
            on=self.decoder(*branches)
            output.update(fused_off=off,fused_on=on)
            if joint_mode=="probe": output["fused_half"]=self.decoder(*branches[:3],.5*branches[3])
            fused=on
        else: fused=self.decoder(*branches[:3],gate*branches[3])
        output.update(fused=fused,joint_gain=gate)
        if reconstruct:
            for index,suffix in enumerate(("ir","vi")):
                output[f"reconstruction_{suffix}"]=self.reconstruct(output[f"feature_{suffix}"],index)
        return output if return_aux else fused
