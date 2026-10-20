"""Fuse a user-supplied paired dataset with a saved RelationFusion checkpoint."""

import argparse
import csv
from pathlib import Path
import time

import torch

from nets import RelationFusion
from utils.checkpoint import load_checkpoint
from utils.settings import ModelConfig, config_from_dict
from utils.dataset import PairedImages
from utils.image import save_fused
from utils.metrics import fusion_metrics
from utils.runtime import resolve_device, write_json

ROOT = Path(__file__).resolve().parent

# ==================== 直接在这里修改测试参数 ====================
CHECKPOINT_PATH = "ckpt/latest.pt"
TEST_DATA_ROOT = "data/LLVIP/test"  # 下含 ir/ 和 vis/（或 vi/）
# 自定义两个目录时，将 TEST_DATA_ROOT 设为 None，再填写下面两项。
IR_DIR = None
VIS_DIR = None
OUTPUT_DIR = "results/LLVIP"  # 同时保存到 gray/ 和 rgb/ 子目录
DEVICE = "cuda"  # 也可指定 "cuda:1"
USE_AMP = False
COMPUTE_METRICS = False
SAVE_DIAGNOSTICS = False
DIAGNOSTICS_LIMIT = 4
JOINT_MODE = "auto"  # auto / off；on/gated 仅用于已训练分支的诊断
# 网络结构直接从 checkpoint 恢复，无需在这里重复填写。
# ==============================================================


def project_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--data-root", help="Paired directory containing ir and vis/vi")
    parser.add_argument("--ir-dir", help="Explicit infrared folder")
    parser.add_argument("--vis-dir", help="Explicit visible folder")
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--joint-mode", choices=("auto", "off", "on", "gated"), default=JOINT_MODE)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=USE_AMP)
    parser.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=COMPUTE_METRICS,
                        help="Also write per-pair luminance metrics")
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=SAVE_DIAGNOSTICS,
                        help="Save relation maps/features for inspection")
    parser.add_argument("--diagnostics-limit", type=int, default=DIAGNOSTICS_LIMIT,
                        help="Number of image pairs with diagnostic maps")
    args = parser.parse_args(argv)
    # 命令行给出数据目录时，整组覆盖顶部路径，避免默认 root 与显式目录冲突。
    if args.data_root is None and args.ir_dir is None and args.vis_dir is None:
        args.data_root, args.ir_dir, args.vis_dir = TEST_DATA_ROOT, IR_DIR, VIS_DIR
    if args.data_root and (args.ir_dir or args.vis_dir):
        parser.error("Use --data-root or explicit directories, not both")
    if args.diagnostics_limit < 1:
        parser.error("--diagnostics-limit must be positive")
    if not args.data_root and not (args.ir_dir and args.vis_dir):
        parser.error("Supply a paired data root or both infrared and visible directories")
    return args


def main(argv=None):
    args = arguments(argv)
    root = project_path(args.data_root) if args.data_root else None
    ir_dir = project_path(args.ir_dir) if args.ir_dir else None
    vi_dir = project_path(args.vis_dir) if args.vis_dir else None
    dataset = PairedImages(root, ir_dir, vi_dir)
    output = project_path(args.output).resolve()
    gray_dir, rgb_dir = output / "gray", output / "rgb"
    source_dirs = {p.parent.resolve() for _, ir, vi in dataset.pairs for p in (ir, vi)}
    if {output, gray_dir.resolve(), rgb_dir.resolve()} & source_dirs:
        raise ValueError("Output directory must not overwrite source images")
    device = resolve_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp inference requires CUDA")
    checkpoint = load_checkpoint(project_path(args.checkpoint))
    if not checkpoint["fusion_trained"]:
        raise ValueError("This checkpoint has not trained the fusion path yet")
    if args.joint_mode == "gated" and not checkpoint["gain_trained"]:
        raise ValueError("The joint gain head has not been trained; use auto/off")
    if args.joint_mode == "on" and checkpoint["model"]["joint_updates"].item() == 0:
        raise ValueError("The joint candidate branch has not been trained")
    model = RelationFusion(config_from_dict(ModelConfig, checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    gray_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, times = [], []
    with torch.inference_mode():
        for index in range(len(dataset)):
            pair = dataset[index]
            ir, vi = [pair[k][None].to(device) for k in ("ir", "vi")]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            collect = args.diagnostics and index < args.diagnostics_limit
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                result = model(ir, vi, return_aux=collect, reconstruct=False, joint_mode=args.joint_mode)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter()-started)
            fused = result["fused"] if collect else result
            image = fused.detach().float().cpu()
            filename = pair["name"] + ".png"
            save_fused(image, gray_dir / filename)
            save_fused(image, rgb_dir / filename, pair["vi_path"], color=True)
            if collect:
                directory = output / "diagnostics"
                directory.mkdir(exist_ok=True)
                maps = {}
                for key, value in result.items():
                    if not isinstance(value, torch.Tensor) or key in ("feature_ir", "feature_vi", "fused"):
                        continue
                    value = value.detach().float().cpu()
                    if value.ndim == 4 and value.shape[1] > 1:
                        key, value = key + "_magnitude", value.abs().mean(1, keepdim=True)
                    maps[key] = value
                torch.save(maps, directory / (pair["name"]+".pt"))
            if args.metrics:
                metric_rows.append({"name": pair["name"], **fusion_metrics(fused, ir, vi)})
            print(f"[{index+1}/{len(dataset)}] {pair['name']}", flush=True)
    summary = {"pairs": len(dataset), "model_version": checkpoint["model_version"],
               "stage": checkpoint["stage"], "gain_trained": checkpoint["gain_trained"], "joint_trained": checkpoint["joint_trained"],
               "joint_mode": args.joint_mode, "checkpoint_epoch": checkpoint["epoch"],
               "gate_blend": float(model.gate_strength), "joint_blend": float(model.joint_strength),
               "reference_snapshot_ready": bool(model.reference_ready),
               "mean_forward_seconds": sum(times)/len(times),
               "timing_note": "Includes first forward; excludes image IO and metrics",
               "gray_directory": str(gray_dir), "rgb_directory": str(rgb_dir),
               "rgb_method": "Fused luminance with visible Cb/Cr chroma"}
    if metric_rows:
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)
        summary["metrics"] = {}
        summary["valid_counts"] = {}
        for key in metric_rows[0]:
            if key == "name":
                continue
            values = [r[key] for r in metric_rows if r[key] is not None]
            summary["metrics"][key] = sum(values)/len(values) if values else None
            summary["valid_counts"][key] = len(values)
    write_json(output / "summary.json", summary)
    print(f"Saved {len(dataset)} grayscale images to {gray_dir}")
    print(f"Saved {len(dataset)} RGB images to {rgb_dir}")


if __name__ == "__main__":
    main()
