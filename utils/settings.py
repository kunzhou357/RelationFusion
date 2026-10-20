"""Internal validated types. User parameters live in train.py and test.py."""
from dataclasses import dataclass, fields
import math

STAGES = ("warmup", "relation", "base", "joint", "calibration")

@dataclass
class ModelConfig:
    dim: int = 64
    common_dim: int = 32
    heads: int = 8
    encoder_blocks: int = 4
    decoder_blocks: int = 4
    expansion: float = 2.
    patch_size: int = 8
    attention_tokens: int = 256
    evidence_scale: float = 16.
    lambda_prior: float = 1.
    interaction: str = "cross"
    verification: bool = True
    decision: bool = True
    compatibility: bool = True
    learned_gain: bool = False
    activation_checkpointing: bool = True

    def __post_init__(self):
        for k in ("dim", "common_dim", "heads", "patch_size", "attention_tokens"):
            if type(getattr(self,k)) is not int or getattr(self,k)<1: raise ValueError(f"{k} must be a positive integer")
        if self.dim % self.heads: raise ValueError("dim must be divisible by heads")
        for k in ("encoder_blocks", "decoder_blocks"):
            if type(getattr(self,k)) is not int or getattr(self,k)<0: raise ValueError(f"{k} must be nonnegative")
        for k in ("expansion", "evidence_scale", "lambda_prior"):
            if not math.isfinite(getattr(self,k)) or getattr(self,k)<=0: raise ValueError(f"Invalid {k}")
        if self.interaction not in ("cross","conv","none"): raise ValueError("Unknown interaction")
        for k in ("verification","decision","compatibility","learned_gain","activation_checkpointing"):
            if type(getattr(self,k)) is not bool: raise ValueError(f"{k} must be boolean")
        if self.learned_gain and self.interaction=="none": raise ValueError("Gain requires interaction")

@dataclass
class TrainConfig:
    data_root: str = "data/MSRS/train"
    output: str = "runs/relation_v3"
    epochs: int = 240
    calibration_epochs: int = 0
    stage_ends: tuple = (20,40,60)
    joint_ramp_epochs: int = 10
    batch_size: int = 8
    crop_size: int = 128
    workers: int = 4
    expected_pairs: int = 1083
    lr: float = 1e-4
    encoder_lr: float = 2e-5
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    clip_grad: float = 1.
    seed: int = 0
    amp: bool = False
    log_every: int = 20
    control_every: int = 2
    control_batch: int = 4
    degrade_probability: float = .5
    noise_max: float = .12
    blur_max: float = 1.8
    stripe_max: float = .12
    contrast_min: float = .3
    source_weight: float = .02
    role_weight: float = 1.
    intervention_weight: float = .1
    support_weight: float = .1
    quality_weight: float = .2
    source_ranking_weight: float = .02
    evidence_weight: float = .1
    compatibility_weight: float = .1
    control_weight: float = 1.
    quality_tau: float = 1.
    quality_intensity_floor: float = .05
    quality_edge_floor: float = .01
    source_contrast_floor: float = .03
    quality_edge_weight: float = .5
    known_gradient_weight: float = 1.
    known_ssim_weight: float = .1
    natural_low_weight: float = .25
    natural_low_sigma: float = 4.
    natural_detail_floor: float = .001
    reference_threshold: float = .7
    trusted_reference: bool = True
    reference_audit_seed: int = 97371
    reference_min_auc: float = .85
    reference_max_ece: float = .15
    reference_min_coverage: float = .9
    gain_tolerance: float = .001

    def __post_init__(self):
        for k in ("epochs","batch_size","crop_size","expected_pairs","log_every","control_every","control_batch","joint_ramp_epochs"):
            if type(getattr(self,k)) is not int or getattr(self,k)<1: raise ValueError(f"{k} must be positive")
        for k in ("workers","calibration_epochs","seed","reference_audit_seed"):
            if type(getattr(self,k)) is not int or getattr(self,k)<0: raise ValueError(f"{k} must be nonnegative")
        if max(self.seed,self.reference_audit_seed)>=2**32: raise ValueError("Seeds must fit uint32")
        self.stage_ends=tuple(self.stage_ends)
        if (len(self.stage_ends)!=3 or any(type(x) is not int for x in self.stage_ends)
                or not 0<self.stage_ends[0]<self.stage_ends[1]<self.stage_ends[2]<self.epochs):
            raise ValueError("stage_ends needs three increasing boundaries below main epochs")
        for f in fields(self):
            v=getattr(self,f.name)
            if isinstance(v,float) and (not math.isfinite(v) or v<0): raise ValueError(f"Invalid {f.name}")
        if not 0<=self.degrade_probability<=1 or not 0<self.contrast_min<=1: raise ValueError("Invalid degradation")
        if not 0<self.reference_threshold<1: raise ValueError("Invalid reference threshold")
        for k in ("reference_min_auc","reference_max_ece","reference_min_coverage"):
            if not 0<getattr(self,k)<1: raise ValueError(f"Invalid {k}")
        if min(self.lr,self.encoder_lr,self.clip_grad,self.quality_tau,self.quality_intensity_floor,
               self.quality_edge_floor,self.source_contrast_floor,self.natural_low_sigma)<=0:
            raise ValueError("Rates/scales must be positive")
        if self.min_lr>min(self.lr,self.encoder_lr): raise ValueError("min_lr exceeds learning rate")
        if type(self.amp) is not bool or type(self.trusted_reference) is not bool: raise ValueError("Expected bool")

    @property
    def total_epochs(self): return self.epochs+self.calibration_epochs

    def stage(self,epoch):
        if not 1<=epoch<=self.total_epochs: raise ValueError("Epoch out of range")
        if epoch>self.epochs: return "calibration"
        return STAGES[sum(epoch>x for x in self.stage_ends)]

    def gate_blend(self,epoch):
        return min(1.,max(0.,(epoch-self.stage_ends[1])/max(1,self.stage_ends[2]-self.stage_ends[1])))

    def joint_blend(self,epoch):
        return min(1.,max(0.,(epoch-self.stage_ends[2])/self.joint_ramp_epochs))

def config_from_dict(cls,values):
    unknown=set(values)-{f.name for f in fields(cls)}
    if unknown: raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**values)
