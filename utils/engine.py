"""Persistent optimizer groups; separate score graphs and continuously trained content."""
import math
import torch
from .augment import degrade_pair
from .controlled_views import make_controls
from .targets import natural_reference,fidelity_target,fidelity_settings


class EpochSchedule:
    """Absolute completed-epoch schedule, including initially inactive groups.
    Never resets Adam moments at a curriculum transition. All states are saved.
    """
    def __init__(self,optimizer,config,group):
        self.optimizer,self.config,self.group=optimizer,config,group
        self.last_epoch=0
        self.apply()

    def rate(self):
        c,g,s=self.config,self.group,self.last_epoch
        start={"encoder":0,"content":0,"relation":c.stage_ends[0],"quality":c.stage_ends[0],
               "decision":c.stage_ends[0],"joint":c.stage_ends[2],"gain":c.epochs}[g]
        end=c.total_epochs if g=="gain" else c.epochs
        if s<start or end<=start: return 0.
        progress=min(1.,max(0.,(s-start)/max(1,end-start)))
        base=c.encoder_lr if g=="encoder" and s>=c.stage_ends[0] else c.lr
        value=c.min_lr+(base-c.min_lr)*.5*(1+math.cos(math.pi*progress))
        if g=="joint": value*=min(1.,(s-start+1)/c.joint_ramp_epochs)
        return value

    def apply(self):
        for p in self.optimizer.param_groups: p["lr"]=self.rate()

    def step(self):
        self.last_epoch+=1;self.apply()

    def state_dict(self): return {"version":1,"group":self.group,"last_epoch":self.last_epoch}

    def load_state_dict(self,state):
        if state.get("version")!=1 or state.get("group")!=self.group: raise ValueError("Schedule mismatch")
        self.last_epoch=state["last_epoch"];self.apply()


def build_optimizers(model,config):
    groups={}
    for name,p in model.named_parameters():
        group=model.parameter_group(name)
        if group is not None: groups.setdefault(group,[]).append(p)
    optimizers={name:torch.optim.AdamW(params,lr=config.lr,weight_decay=config.weight_decay)
                for name,params in groups.items()}
    schedulers={name:EpochSchedule(opt,config,name) for name,opt in optimizers.items()}
    return optimizers,schedulers


def restore_optimizers(optimizers,schedulers,checkpoint,config):
    if set(optimizers)!=set(checkpoint["optimizers"]) or set(schedulers)!=set(checkpoint["schedulers"]):
        raise ValueError("Optimizer groups changed")
    for name,opt in optimizers.items():
        opt.load_state_dict(checkpoint["optimizers"][name])
        schedulers[name].load_state_dict(checkpoint["schedulers"][name])
        if schedulers[name].last_epoch!=checkpoint["epoch"]: raise ValueError("Schedule/epoch mismatch")


def diagnostics(output):
    result={}
    with torch.no_grad():
        for key in ("support","novelty_ir","novelty_vi","quality_common","quality_private_ir","quality_private_vi",
                    "source_quality_ir","source_quality_vi","weight_vi","uncertainty","compatibility","joint_gain","predicted_gain"):
            if key in output:
                x=output[key].float();result[key+"_mean"]=float(x.mean());result[key+"_std"]=float(x.std(unbiased=False))
        for key in ("shared","private_ir","private_vi","unexplained_ir","unexplained_vi","joint"):
            if key in output:
                x=output[key].float();result[key+"_rms"]=float(x.square().mean().sqrt())
    return result


def content_gradient_probe(model,fusion,role):
    params=[model.decomposition.common_project.weight,model.decomposition.private_content_ir[-1].weight,
            model.decomposition.private_content_vi[-1].weight]
    vectors=[]
    for loss in (fusion,role):
        gradients=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
        vectors.append(torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten() for p,g in zip(params,gradients)]))
    a,b=vectors;den=a.norm()*b.norm()
    values={"probe_fusion_gradient":float(a.norm()),"probe_role_gradient":float(b.norm()),
            "probe_gradient_cosine":float((a@b)/den) if den>1e-12 else None}
    return {k:v if v is not None and math.isfinite(v) else None for k,v in values.items()}


def train_step(model,batch,objectives,optimizers,scaler,config,device,global_step=0,collect_diagnostics=False):
    model.train();stage=model.stage
    ir,vi=[batch[k].to(device,non_blocking=True) for k in ("ir","vi")]
    for optimizer in optimizers.values(): optimizer.zero_grad(set_to_none=True)
    enabled=config.amp and device.type=="cuda"
    control_count=0;probe={};role_total=None;target_stats={}
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=enabled):
        if stage=="warmup":
            output=model.warmup(ir,vi);losses={"source":objectives.source(output,ir,vi)};total=losses["source"]
        elif stage=="calibration":
            # Extra calibration uses known-target observations derived from training
            # source textures. It never labels the natural weak fusion prior as GT.
            control=make_controls(config.control_batch,*ir.shape[-2:],device,config,(ir,vi))
            output=model(control["ir"],control["vi"],return_aux=True,joint_mode="probe")
            losses={"gain":objectives.gain(output,control)};total=losses["gain"];control_count=config.control_batch
        else:
            data=degrade_pair(ir,vi,config)
            trust=model.reference_trust(data["reference_ir"],data["reference_vi"])
            data["proxy"]=natural_reference(data["reference_ir"],data["reference_vi"],trust,
                config.reference_threshold,config.natural_low_sigma,config.natural_low_weight,config.natural_detail_floor)
            target_stats.update({"reference_"+k+"_fraction":float(data["proxy"][k].mean())
                                 for k in ("valid_x","valid_y","low_valid","exact_x","exact_y")})
            mode="dual" if stage=="joint" and model.config.interaction!="none" else "off"
            output=model(data["ir"],data["vi"],return_aux=True,joint_mode=mode)
            natural=objectives.fusion(output,data)
            losses={"fusion":natural};total=natural
            if config.source_weight:
                source=model.warmup(data["reference_ir"],data["reference_vi"])
                losses["source"]=objectives.source(source,data["reference_ir"],data["reference_vi"])
                total=total+config.source_weight*losses["source"]
            losses["source_ranking"]=objectives.natural_source_ranking(model,data,output)
            total=total+config.source_ranking_weight*losses["source_ranking"]
            if (global_step+1)%config.control_every==0:
                control=make_controls(config.control_batch,*ir.shape[-2:],device,config,(ir,vi))
                controlled=model(control["ir"],control["vi"],return_aux=True,joint_mode=mode)
                cf=objectives.fusion(controlled,control)
                key=("without_common","without_ir","without_vi")[(global_step//config.control_every)%3]
                roles=objectives.roles(model,control,key)
                scores=objectives.scores(model,controlled,control)
                weighted={k:getattr(config,k+"_weight")*v for k,v in {**roles,**scores}.items()}
                role_total=sum(weighted[k] for k in roles)
                total=total+config.control_weight*(cf+sum(weighted.values()))
                losses.update(control_fusion=cf,**{"control_"+k:v for k,v in {**roles,**scores}.items()})
                control_count=config.control_batch
                if collect_diagnostics:
                    probe=content_gradient_probe(model,natural+config.control_weight*cf,config.control_weight*role_total)
                    for name,(target,valid) in objectives.candidate_targets(controlled,control).items():
                        chosen=valid.bool();values=target[chosen]
                        target_stats["quality_target_"+name+"_valid_fraction"]=float(valid.mean())
                        if len(values):
                            target_stats["quality_target_"+name+"_mean"]=float(values.mean())
                            target_stats["quality_target_"+name+"_low_fraction"]=float((values<.3).float().mean())
                            target_stats["quality_target_"+name+"_high_fraction"]=float((values>.8).float().mean())
    if not all(bool(torch.isfinite(x).all()) for x in (*losses.values(),total)):
        raise FloatingPointError("Nonfinite objective: no optimizer was stepped")
    scaler.scale(total).backward()
    active=[];norms={};finite=True
    for name,opt in optimizers.items():
        parameters=[p for group in opt.param_groups for p in group["params"] if p.grad is not None]
        if not parameters: continue
        scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(parameters,config.clip_grad,error_if_nonfinite=False)
        value=float(norm);finite=finite and math.isfinite(value)
        norms[name+"_grad_norm"]=value if math.isfinite(value) else None
        active.append((name,opt))
    if finite:
        for _,opt in active: scaler.step(opt)
    elif not enabled:
        raise FloatingPointError("Nonfinite gradient: all optimizer groups were skipped")
    scaler.update()
    if finite:
        if stage not in ("warmup","calibration"): model.fusion_updates.add_(1)
        if stage=="joint" and model.config.interaction!="none": model.joint_updates.add_(1)
        if stage=="calibration": model.gain_updates.add_(1)
    record={k:float(v.detach()) for k,v in {**losses,"total":total}.items()}
    record.update(norms);record.update(probe);record.update(target_stats)
    record.update(optimizer_step=int(finite),gradient_overflow=int(not finite),control_pairs=control_count,
        natural_pairs=0 if stage=="calibration" else len(ir),visited_pairs=len(ir),amp_scale=float(scaler.get_scale()),
        gate_blend=float(model.gate_strength),joint_blend=float(model.joint_strength),
        trusted_reference_active=int(bool(model.reference_ready)))
    if collect_diagnostics: record.update(diagnostics(output))
    return record


def binary_auc(scores,labels):
    labels=labels.bool().flatten();scores=scores.float().flatten()
    npos=int(labels.sum());nneg=len(labels)-npos
    if not npos or not nneg: return None
    values,order=scores.sort();_,inverse,counts=torch.unique_consecutive(values,return_inverse=True,return_counts=True)
    end=counts.cumsum(0).float();rank=end-(counts.float()-1)/2
    return float((rank[inverse][labels[order]].sum()-npos*(npos+1)/2)/(npos*nneg))


@torch.no_grad()
def audit_reference_teacher(model,config,device):
    """Fixed procedural DEVELOPMENT guard, not the final held-out mechanism test.
    Restores RNG state and never selects a fusion checkpoint by test metrics.
    """
    from dataclasses import replace
    devices=list(range(torch.cuda.device_count())) if device.type=="cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.reference_audit_seed)
        c=replace(config,degrade_probability=1.,noise_max=.2,stripe_max=.2,blur_max=2.5)
        data=make_controls(12,64,64,device,c)
        qs=[];ts=[]
        for s in ("ir","vi"):
            q=model.verification.source(data[s])
            source_reference=data.get("source_reference_"+s,data["reference_"+s])
            t,v=fidelity_target(data[s],source_reference,source=True,**fidelity_settings(config))
            mask=(v*data["source_absolute_valid"]).bool();qs.append(q[mask]);ts.append(t[mask])
        q,t=torch.cat(qs),torch.cat(ts)
        extremes=(t>.8)|(t<.3)
        auc=binary_auc(q[extremes],t[extremes]>.8)
        mse=float((q-t).square().mean());constant=float((t-t.mean()).square().mean())
        ece=0.
        for index in range(10):
            mask=(q>=index/10)&((q<(index+1)/10) if index<9 else q<=1)
            if mask.any(): ece+=float(mask.float().mean()*(q[mask].mean()-t[mask].mean()).abs())
        positive=t>.9
        coverage=float((q[positive]>=config.reference_threshold).float().mean()) if positive.any() else 0.
        passed=bool(auc is not None and auc>config.reference_min_auc and mse<constant
                    and ece<config.reference_max_ece and coverage>config.reference_min_coverage)
    model.reference_attempted.fill_(True)
    if passed: model.freeze_reference()
    return {"passed":passed,"auc":auc,"mse":mse,"constant_mean_mse":constant,"ece":ece,
            "high_quality_coverage":coverage,"seed":config.reference_audit_seed,
            "scope":"procedural development source-fidelity guard; not natural-image validation"}
