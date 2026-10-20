"""Train v3 on ALL 1083 MSRS pairs. No validation split or best checkpoint."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from nets import RelationFusion
from utils.settings import ModelConfig,TrainConfig,config_from_dict
from utils.checkpoint import load_checkpoint,checkpoint_payload,save_epoch,restore_rng
from utils.engine import build_optimizers,restore_optimizers,train_step,audit_reference_teacher
from utils.losses import FusionObjectives
from utils.dataset import PairedImages
from utils.runtime import seed_all,seed_worker,resolve_device,write_json,append_json

ROOT=Path(__file__).resolve().parent
# ==================== 直接在这里修改参数，无需配置文件 ====================
DEVICE="cuda"
RESUME_CHECKPOINT=None
CHECK_DATA_ONLY=False
TRAIN_PARAMETERS=dict(
    data_root="data/MSRS/train",output="runs/relation_v3",expected_pairs=1083,
    epochs=240,calibration_epochs=0,stage_ends=(20,40,60),joint_ramp_epochs=10,
    batch_size=4,crop_size=128,workers=4,lr=1e-4,encoder_lr=2e-5,min_lr=1e-6,
    weight_decay=1e-4,clip_grad=1.,seed=0,amp=False,log_every=20,
    control_every=2,control_batch=4,degrade_probability=.5,
    noise_max=.12,blur_max=1.8,stripe_max=.12,contrast_min=.3,
    source_weight=.02,role_weight=1.,intervention_weight=.1,support_weight=.1,
    quality_weight=.2,source_ranking_weight=.02,evidence_weight=.1,compatibility_weight=.1,control_weight=1.,
    quality_tau=1.,quality_intensity_floor=.05,quality_edge_floor=.01,source_contrast_floor=.03,quality_edge_weight=.5,
    known_gradient_weight=1.,known_ssim_weight=.1,natural_low_weight=.25,natural_low_sigma=4.,natural_detail_floor=.001,
    reference_threshold=.7,trusted_reference=True,reference_audit_seed=97371,
    reference_min_auc=.85,reference_max_ece=.15,reference_min_coverage=.9,gain_tolerance=.001,
)
MODEL_PARAMETERS=dict(
    dim=64,common_dim=32,heads=8,encoder_blocks=4,decoder_blocks=4,expansion=2.,
    patch_size=8,attention_tokens=256,evidence_scale=16.,lambda_prior=1.,
    interaction="cross",verification=True,decision=True,compatibility=True,
    learned_gain=False,  # 可选校准：同时设置True和calibration_epochs=20，主体仍为240轮。
    activation_checkpointing=True,  # 用重算减少持续反向传播的激活显存，推理时不启用。
)
RESUME_OVERRIDES={}  # 仅允许data_root/output/workers/amp/log_every等运行参数。
# ======================================================================


def project_path(value):
    path=Path(value).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",type=Path,default=RESUME_CHECKPOINT)
    parser.add_argument("--device",default=DEVICE)
    parser.add_argument("--check-data",action=argparse.BooleanOptionalAction,default=CHECK_DATA_ONLY)
    parser.add_argument("--stage-ends",type=int,nargs=3)
    for key,value in {**TRAIN_PARAMETERS,**MODEL_PARAMETERS}.items():
        if key=="stage_ends": continue
        option="--"+key.replace("_","-")
        if isinstance(value,bool): parser.add_argument(option,action=argparse.BooleanOptionalAction,default=None)
        else: parser.add_argument(option,type=type(value),default=None)
    return parser.parse_args(argv)


def settings(args,checkpoint):
    base_model=dict(checkpoint["model_config"] if checkpoint else MODEL_PARAMETERS)
    base_train=dict(checkpoint["train_config"] if checkpoint else TRAIN_PARAMETERS)
    model_values,train_values=dict(base_model),dict(base_train)
    allowed={"data_root","output","workers","amp","log_every"}
    if checkpoint:
        if set(RESUME_OVERRIDES)-allowed: raise ValueError("Unsupported resume override")
        train_values.update(RESUME_OVERRIDES)
    for values in (model_values,train_values):
        for key in values:
            if getattr(args,key,None) is not None: values[key]=getattr(args,key)
    train_values["stage_ends"]=tuple(train_values["stage_ends"])
    base_train["stage_ends"]=tuple(base_train["stage_ends"])
    if checkpoint:
        if model_values!=base_model: raise ValueError("Resume cannot change architecture")
        if any(train_values[k]!=base_train[k] for k in base_train if k not in allowed):
            raise ValueError("Resume must keep recipe/horizon. Set main epochs before starting a new run.")
        if args.output is None and "output" not in RESUME_OVERRIDES:
            train_values["output"]=str(project_path(args.resume).parent.parent)
    mc,tc=config_from_dict(ModelConfig,model_values),config_from_dict(TrainConfig,train_values)
    if mc.learned_gain!=bool(tc.calibration_epochs):
        raise ValueError("learned_gain and calibration_epochs>0 must be enabled together")
    for key in ("data_root","output"): setattr(tc,key,str(project_path(getattr(tc,key))))
    return mc,tc


def main(argv=None):
    args=arguments(argv)
    if args.resume: args.resume=project_path(args.resume)
    checkpoint=load_checkpoint(args.resume,for_resume=True) if args.resume else None
    mc,config=settings(args,checkpoint);seed_all(config.seed)
    dataset=PairedImages(config.data_root,crop_size=config.crop_size,augment=True,expected_pairs=config.expected_pairs)
    manifest=dataset.manifest()
    if args.check_data:
        print({"pairs":len(dataset),"shape":list(dataset[0]["ir"].shape),"all_pairs_used":True,"model_version":"relation_v3"})
        return
    device=resolve_device(args.device)
    if config.amp and device.type!="cuda": raise ValueError("Training AMP requires CUDA")
    directory=Path(config.output)
    if not checkpoint and directory.exists() and any(directory.iterdir()): raise FileExistsError("Run exists; use --resume or a new output")
    directory.mkdir(parents=True,exist_ok=True)
    generator=torch.Generator().manual_seed(config.seed)
    loader=DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.workers,
        drop_last=False,pin_memory=device.type=="cuda",worker_init_fn=seed_worker,generator=generator,persistent_workers=False)
    model=RelationFusion(mc).to(device);objectives=FusionObjectives(config).to(device)
    start,global_step=1,0
    if checkpoint:
        if checkpoint["data_fingerprint"]!=manifest["fingerprint"]: raise ValueError("Dataset changed since checkpoint")
        model.load_state_dict(checkpoint["model"],strict=True)
        start,global_step=checkpoint["epoch"]+1,checkpoint["global_step"]
    if start>config.total_epochs: raise ValueError("This run has already completed its recipe")
    model.configure_epoch(start,config)
    optimizers,schedulers=build_optimizers(model,config)
    scaler=torch.amp.GradScaler("cuda",enabled=config.amp)
    if checkpoint:
        restore_optimizers(optimizers,schedulers,checkpoint,config)
        if config.amp and checkpoint["scaler"]: scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"],generator)
    write_json(directory/"run_info.json",{"model_version":"relation_v3","model":asdict(mc),"train":asdict(config),
        "total_epochs":config.total_epochs,"parameters":sum(p.numel() for p in model.parameters())})
    write_json(directory/"data_manifest.json",manifest)
    append_json(directory/"events.jsonl",{"event":"resume" if checkpoint else "start","utc":datetime.now(timezone.utc).isoformat(),
        "start_epoch":start,"pairs":len(dataset),"torch":torch.__version__,"device":str(device)})
    announced=None
    for epoch in range(start,config.total_epochs+1):
        model.configure_epoch(epoch,config)
        if model.stage!=announced:
            if model.stage=="calibration" and not model.joint_updates.item(): raise RuntimeError("Cannot calibrate untrained joint branch")
            append_json(directory/"events.jsonl",{"event":"stage","epoch":epoch,"stage":model.stage,
                "trainable":[n for n,p in model.named_parameters() if p.requires_grad]})
            announced=model.stage
        sums,counts={},{};controls=updates=0;started=time.perf_counter()
        for index,batch in enumerate(loader):
            collect=(index+1)%config.log_every==0 or index+1==len(loader)
            record=train_step(model,batch,objectives,optimizers,scaler,config,device,global_step,collect)
            global_step+=1;size=len(batch["ir"]);controls+=record["control_pairs"];updates+=record["optimizer_step"]
            for key,value in record.items():
                if value is not None: sums[key]=sums.get(key,0)+value*size;counts[key]=counts.get(key,0)+size
            if collect:
                append_json(directory/"steps.jsonl",{"epoch":epoch,"step":global_step,"stage":model.stage,**record})
                print(f"epoch {epoch:03d}/{config.total_epochs} {model.stage} {index+1}/{len(loader)} loss={record['total']:.5f}",flush=True)
        if (epoch>=config.stage_ends[2] and config.trusted_reference and mc.verification and not bool(model.reference_attempted)):
            audit=audit_reference_teacher(model,config,device)
            append_json(directory/"events.jsonl",{"event":"reference_development_guard","epoch":epoch,**audit})
            print("Trusted reference snapshot:","enabled" if audit["passed"] else "rejected; unmasked weak baseline retained",flush=True)
        rates={name:opt.param_groups[0]["lr"] for name,opt in optimizers.items()}
        for scheduler in schedulers.values(): scheduler.step()
        append_json(directory/"epochs.jsonl",{"epoch":epoch,"stage":model.stage,"images":len(dataset),
            "control_images":controls,"successful_updates":updates,"seconds":time.perf_counter()-started,
            "learning_rates":rates,**{k:sums[k]/counts[k] for k in sums}})
        if epoch%10==0 or epoch==config.total_epochs:
            payload=checkpoint_payload(model,mc,config,optimizers,schedulers,scaler,epoch,global_step,manifest,generator)
            save_epoch(directory/"checkpoints",payload,final=epoch==config.total_epochs)
    print("Training complete:",directory/"checkpoints/latest.pt",flush=True)


if __name__=="__main__": main()
