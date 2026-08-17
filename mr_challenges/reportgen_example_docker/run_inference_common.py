"""Shared helpers for the MR-RATE baselines (classification + report gen)."""
import os
import shutil
import threading
import zipfile
import numpy as np
import torch
from pathlib import Path

from data import load_and_resample_nii, crop_or_pad  # upstream, on sys.path

TARGET_SPACING = (1.0, 0.5, 0.5)
TARGET_SHAPE = (256, 384, 384)
POSTERIOR_SHIFT_MM = 15.0
MAX_TOKENS_PER_STUDY = 4096


def evenly_spaced_indices(total: int, limit: int) -> np.ndarray:
    if limit <= 0 or total <= limit:
        return np.arange(total)
    return np.unique(np.linspace(0, total - 1, num=limit).round().astype(np.int64))


class _LazyZipAtlasPaths:
    """Extracts one study zip's atlas volumes on demand, prefetching the next.

    The platform mounts phase data as-is via GCS FUSE, so /input contains
    2029 unextracted <STUDY>.zip files (each: <STUDY>/{img,atlas,seg}/...).
    Extracting everything up front (~114 GB of atlas volumes alone) would
    overflow the 200 GB boot disk beside the image and extracted weights.
    Instead, when study i's paths are first iterated:
      - study i extracts if a prefetch hasn't already done it (thread-safe),
      - study i+1 starts extracting on a daemon thread, so the FUSE read +
        inflate overlaps the GPU's work on study i instead of idling it,
      - study i-2 is deleted (i-1 may still be open by the caller).
    Disk high-water mark: 3 studies (~170 MB). Quacks like the list of
    Paths that discover_studies used to return.
    """

    def __init__(self, zip_path: Path, scratch: Path):
        self.zip_path = zip_path
        self.scratch = scratch
        self._dest = scratch / zip_path.stem
        self._paths = None
        self._lock = threading.Lock()
        self._prev = None   # linked by discover_studies for cleanup ordering
        self._next = None   # linked by discover_studies for prefetch

    def _extract(self):
        """Idempotent, thread-safe extraction of this study's atlas members."""
        with self._lock:
            if self._paths is None:
                with zipfile.ZipFile(self.zip_path) as z:
                    members = [m for m in z.namelist()
                               if "/atlas/" in m and m.endswith((".nii.gz", ".nii"))]
                    z.extractall(self._dest, members)
                self._paths = (sorted(self._dest.rglob("*.nii.gz"))
                               + sorted(self._dest.rglob("*.nii")))
        return self._paths

    def _free(self):
        shutil.rmtree(self._dest, ignore_errors=True)

    def _materialize(self):
        paths = self._extract()  # no-op if the prefetch thread already ran
        if self._next is not None:
            threading.Thread(target=self._next._extract, daemon=True).start()
        if self._prev is not None and self._prev._prev is not None:
            self._prev._prev._free()
        return paths

    def __iter__(self):
        return iter(self._materialize())

    def __len__(self):
        return len(self._materialize())


def discover_studies(root: Path):
    """Challenge layout: /input/<STUDY>.zip (platform) or pre-extracted
    /input/<STUDY>/atlas/*.nii.gz (local tests). Zip entries extract lazily,
    one study at a time — see _LazyZipAtlasPaths."""
    zips = sorted(root.glob("*.zip"))
    if zips:
        scratch = Path(os.environ.get("FORITHMUS_SCRATCH", "/tmp/forithmus-input"))
        scratch.mkdir(parents=True, exist_ok=True)
        entries = [_LazyZipAtlasPaths(z, scratch) for z in zips]
        # Chain entries so each study prefetches its successor and frees the
        # study two behind it (see _LazyZipAtlasPaths._materialize).
        for a, b in zip(entries, entries[1:]):
            b._prev = a
            a._next = b
        return [(e.zip_path.stem, e) for e in entries]

    out = []
    for study_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        atlas = study_dir / "atlas"
        if not atlas.is_dir():
            alt = study_dir / "atlas_space" / "img"
            atlas = alt if alt.is_dir() else None
        if atlas is None:
            continue
        vols = sorted(atlas.glob("*.nii.gz")) + sorted(atlas.glob("*.nii"))
        if vols:
            out.append((study_dir.name, vols))
    return out


def load_study_stack(paths, normalizer) -> torch.Tensor:
    """Mirror MRReportDatasetInfer exactly: resample -> .normalize() ->
    crop_or_pad (float32 [D,H,W]) -> bf16 [1,D,H,W] -> stack [N,1,D,H,W]."""
    shift_vox = int(round(POSTERIOR_SHIFT_MM / TARGET_SPACING[2]))
    tensors = []
    for p in paths:
        vol = load_and_resample_nii(str(p), TARGET_SPACING)
        vol = normalizer.normalize(vol)
        arr = crop_or_pad(vol, TARGET_SHAPE, shift_vox)
        tensors.append(torch.from_numpy(arr).unsqueeze(0).to(torch.bfloat16))
    return torch.stack(tensors, dim=0)  # [N,1,D,H,W]
