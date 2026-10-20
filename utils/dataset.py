"""File-name paired datasets; every training pair is retained."""

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .image import EXTENSIONS, load_luminance


def image_index(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in result:
                raise ValueError(f"Duplicate image stem {path.stem!r} in {folder}")
            result[path.stem] = path
    if not result:
        raise ValueError(f"No supported images in {folder}")
    return result


def paired_paths(root=None, ir_dir=None, vi_dir=None):
    if root is not None and (ir_dir is not None or vi_dir is not None):
        raise ValueError("Supply a paired root or explicit directories, not both")
    if root is not None:
        root = Path(root)
        ir_dir = root / "ir"
        candidates = [root / name for name in ("vis", "vi") if (root / name).is_dir()]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one vis/vi directory below {root}")
        vi_dir = candidates[0]
    if ir_dir is None or vi_dir is None:
        raise ValueError("Supply a paired root or both ir_dir and vi_dir")
    ir, vi = image_index(ir_dir), image_index(vi_dir)
    if set(ir) != set(vi):
        raise ValueError(f"Pair mismatch: IR-only {sorted(set(ir)-set(vi))[:8]}, "
                         f"visible-only {sorted(set(vi)-set(ir))[:8]}")
    return [(name, ir[name], vi[name]) for name in sorted(ir)]


class PairedImages(Dataset):
    def __init__(self, root=None, ir_dir=None, vi_dir=None, crop_size=None,
                 augment=False, expected_pairs=None):
        self.pairs = paired_paths(root, ir_dir, vi_dir)
        if expected_pairs is not None and len(self.pairs) != expected_pairs:
            raise ValueError(f"Expected all {expected_pairs} pairs, found {len(self.pairs)}")
        self.crop_size, self.augment = crop_size, augment

    def __len__(self):
        return len(self.pairs)

    def manifest(self):
        rows = [{"name": n, "ir": str(i.resolve()), "visible": str(v.resolve()),
                 "ir_bytes": i.stat().st_size, "visible_bytes": v.stat().st_size}
                for n, i, v in self.pairs]
        portable = [{k: row[k] for k in ("name", "ir_bytes", "visible_bytes")} for row in rows]
        fingerprint = hashlib.sha256(json.dumps(portable, sort_keys=True).encode()).hexdigest()
        return {"count": len(rows), "fingerprint": fingerprint,
                "fingerprint_kind": "paired names and file sizes, not content hashes", "pairs": rows}

    def __getitem__(self, index):
        name, ir_path, vi_path = self.pairs[index]
        ir, vi = load_luminance(ir_path), load_luminance(vi_path, visible=True)
        if ir.shape != vi.shape:
            raise ValueError(f"Image pair {name} has different shapes: {ir.shape}, {vi.shape}")
        if self.crop_size:
            size = self.crop_size
            dh, dw = max(0, size - ir.shape[-2]), max(0, size - ir.shape[-1])
            if dh or dw:
                ir, vi = [F.pad(x, (0, dw, 0, dh), mode="replicate") for x in (ir, vi)]
            y = int(torch.randint(ir.shape[-2] - size + 1, ()).item())
            x = int(torch.randint(ir.shape[-1] - size + 1, ()).item())
            ir, vi = [t[:, y:y+size, x:x+size] for t in (ir, vi)]
        if self.augment:
            turns = int(torch.randint(4, ()).item())
            ir, vi = [t.rot90(turns, (-2, -1)) for t in (ir, vi)]
            if torch.rand(()) < .5:
                ir, vi = ir.flip(-1), vi.flip(-1)
        return {"ir": ir.contiguous(), "vi": vi.contiguous(), "name": name,
                "ir_path": str(ir_path), "vi_path": str(vi_path)}
