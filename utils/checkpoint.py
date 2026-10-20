"""Atomic snapshots, full resume state, and a fixed ten-epoch save policy."""

from dataclasses import asdict
from pathlib import Path
import os
import random

import numpy as np
import torch


def rng_state(generator=None):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": generator.get_state() if generator is not None else None}


def restore_rng(state, generator=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        # A cloud resume may expose fewer GPUs than the original host.
        for index, saved in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(saved.cpu(), device=index)
    if generator is not None and state.get("loader") is not None:
        generator.set_state(state["loader"].cpu())


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def checkpoint_payload(model,model_config,train_config,optimizers,schedulers,scaler,
                       epoch,global_step,manifest,generator):
    return {"format_version":3,"model_version":"relation_v3","stage":model.stage,
        "joint_trained":bool(model.joint_updates.item()>0),"gain_trained":bool(model.gain_updates.item()>0),
        "fusion_trained":bool(model.fusion_updates.item()>0),
        "model_config":asdict(model_config),"train_config":asdict(train_config),"model":model.state_dict(),
        "optimizers":{k:o.state_dict() for k,o in optimizers.items()},
        "schedulers":{k:s.state_dict() for k,s in schedulers.items()},"scaler":scaler.state_dict(),
        "epoch":epoch,"global_step":global_step,"data_fingerprint":manifest["fingerprint"],"rng":rng_state(generator)}


def save_epoch(directory,payload,final=False):
    epoch=payload["epoch"];paths=[]
    if epoch%10==0:
        path=Path(directory)/f"epoch_{epoch:04d}.pt";atomic_save(path,payload);paths.append(path)
    if epoch%10==0 or final:
        path=Path(directory)/"latest.pt";atomic_save(path,payload);paths.append(path)
    return paths


def load_checkpoint(path,for_resume=False):
    value=torch.load(path,map_location="cpu",weights_only=False)
    if not isinstance(value,dict) or value.get("format_version")!=3 or value.get("model_version")!="relation_v3":
        raise ValueError("Expected relation_v3 checkpoint. v1/v2 need their archived source; no strict=False migration.")
    for key in ("model_config","train_config","model","epoch","global_step","stage"):
        if key not in value: raise ValueError(f"Checkpoint missing {key}")
    for key in ("model_config","train_config","model"):
        if not isinstance(value[key],dict): raise ValueError(f"{key} must be a dictionary")
    for key in ("epoch","global_step"):
        if type(value[key]) is not int or value[key]<1: raise ValueError(f"Invalid {key}")
    from .settings import TrainConfig,ModelConfig,config_from_dict
    config=config_from_dict(TrainConfig,value["train_config"])
    model_config=config_from_dict(ModelConfig,value["model_config"])
    if model_config.learned_gain!=bool(config.calibration_epochs): raise ValueError("Calibration recipe/architecture mismatch")
    if config.stage(value["epoch"])!=value["stage"]: raise ValueError("Stage disagrees with epoch/recipe")
    for key,flag in (("fusion_updates","fusion_trained"),("joint_updates","joint_trained"),("gain_updates","gain_trained")):
        n=value["model"].get(key)
        if not isinstance(n,torch.Tensor) or n.numel()!=1 or n.dtype!=torch.long or n.item()<0: raise ValueError(f"Invalid {key}")
        if value.get(flag)!=bool(n.item()>0): raise ValueError(f"Invalid training provenance: {flag}")
    if value["gain_trained"] and (not value["joint_trained"] or value["stage"]!="calibration"):
        raise ValueError("Injection policy has invalid provenance")
    if value["joint_trained"] and not value["fusion_trained"]: raise ValueError("Joint branch requires trained base")
    for key in ("gate_strength","joint_strength"):
        x=value["model"].get(key)
        if not isinstance(x,torch.Tensor) or x.numel()!=1 or not 0<=float(x)<=1: raise ValueError(f"Invalid {key}")
    for key in ("reference_ready","reference_attempted"):
        x=value["model"].get(key)
        if not isinstance(x,torch.Tensor) or x.numel()!=1 or x.dtype!=torch.bool: raise ValueError(f"Invalid {key}")
    if bool(value["model"]["reference_ready"]) and not bool(value["model"]["reference_attempted"]):
        raise ValueError("Reference snapshot has no development audit")
    if for_resume:
        for key in ("optimizers","schedulers","scaler","rng","data_fingerprint"):
            if key not in value: raise ValueError(f"Resume missing {key}")
        for key in ("optimizers","schedulers","scaler","rng"):
            if not isinstance(value[key],dict): raise ValueError(f"Invalid resume {key}")
        if not {"python","numpy","torch","loader"}<=value["rng"].keys() or value["rng"]["loader"] is None:
            raise ValueError("Resume needs complete random/loader state")
    return value
