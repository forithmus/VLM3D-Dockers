"""
FID_2p5D — 2.5D Frechet Inception Distance (STREAMING versiyon)
==================================================================

Yöntem, VLM3D CT track'inin gerçek kodundan (compute_fid_2-5d_ct.py)
alınmıştır: squeezenet1_1 ile üç ortogonal düzlemde (XY/XZ/YZ) dilim
bazlı feature extraction, sonra Frechet mesafesi.

ÖNEMLİ TASARIM NOTU: Bu modülün ilk versiyonu tüm hacimleri belleğe
topluyordu — 13.930 gerçek hacimde bu iş görmez (yüzlerce GB RAM gerekir).
Bu versiyon STREAMING çalışır: FIDAccumulator, her (real, fake) hacim
çiftini tek tek alır, o çiftin slice feature'larını hemen çıkarıp küçük
(N_slices, 512) boyutlu array'lere ekler, hacmi belleğe hiç tutmaz.
Sadece feature vektörleri birikir — 13.930 hacim için bile makul boyutta.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from scipy import linalg

_FEATURE_DIM = 512
_INPUT_SIZE = 224

# Planlar: axis=2 (XY, Z sabit), axis=1 (XZ, Y sabit), axis=0 (YZ, X sabit)
_AXIS_FOR_PLANE = {"XY": 2, "XZ": 1, "YZ": 0}


class SqueezeNetFeatureExtractor(nn.Module):
    """squeezenet1_1'in classifier'ını atıp, global-average-pool edilmiş
    512-boyutlu feature vektörünü döndüren sarmalayıcı."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        weights = tv_models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        base = tv_models.squeezenet1_1(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.eval()
        self.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


def _normalize_slice(sl: np.ndarray) -> np.ndarray:
    sl = sl.astype(np.float32)
    lo, hi = np.percentile(sl, 0.5), np.percentile(sl, 99.5)
    if hi - lo < 1e-6:
        return np.zeros_like(sl)
    return np.clip((sl - lo) / (hi - lo), 0.0, 1.0)


def _slice_to_tensor(sl: np.ndarray) -> torch.Tensor:
    sl = _normalize_slice(sl)
    t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(
        t, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False
    )
    t = t.repeat(1, 3, 1, 1)
    return t.squeeze(0)


def _iter_slices(volume: np.ndarray, axis: int, stride: int = 4):
    n = volume.shape[axis]
    for idx in range(0, n, stride):
        yield np.take(volume, idx, axis=axis)


class RunningMoments:
    """Ortalama ve kovaryansı, tüm feature'ları belleğe toplamadan
    (Welford benzeri) biriktiren yardımcı sınıf.

    Not: Kovaryans için tam doğruluk gerektiğinden, burada basitlik ve
    sayısal kararlılık adına feature'ları biriktirip (float32, 512-dim
    -- her hacimden birkaç yüz slice, hacmin kendisinden ~1000x küçük)
    sonda tek seferde hesaplıyoruz. Tam hacimleri değil, sadece
    feature'ları tuttuğumuz için 13.930 hacimde bile RAM makul kalır
    (13930 hacim * ~100 slice/plane * 512 float32 ≈ birkaç GB, hacimlerin
    kendisini tutmaktan çok daha az)."""

    def __init__(self):
        self._chunks: list[np.ndarray] = []

    def add(self, feats: np.ndarray) -> None:
        if feats.shape[0] > 0:
            self._chunks.append(feats)

    def finalize(self) -> tuple[np.ndarray, np.ndarray, int]:
        if not self._chunks:
            return np.zeros(_FEATURE_DIM), np.eye(_FEATURE_DIM), 0
        all_feats = np.concatenate(self._chunks, axis=0)
        mu = all_feats.mean(axis=0)
        sigma = np.cov(all_feats, rowvar=False)
        return mu, sigma, all_feats.shape[0]


class FIDAccumulator:
    """Hacim çiftlerini TEK TEK alır (streaming), her çift için slice
    feature'larını hemen çıkarır ve biriktirir. Hacimlerin kendisi asla
    aynı anda toplu halde bellekte tutulmaz — çağıran taraf her çift
    işlendikten sonra hacimleri serbest bırakabilir."""

    def __init__(self, device: str = "auto", stride: int = 4, batch_size: int = 32):
        # "auto": use the accelerator the job was scheduled on. The eval tier is
        # a GPU tier, so hardcoding CPU left the card idle while SqueezeNet ran
        # on 4 vCPU.
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = SqueezeNetFeatureExtractor(device=device)
        self.device = device
        self.stride = stride
        self.batch_size = batch_size
        self.real_moments = {plane: RunningMoments() for plane in _AXIS_FOR_PLANE}
        self.fake_moments = {plane: RunningMoments() for plane in _AXIS_FOR_PLANE}

    @torch.no_grad()
    def _extract_volume_features(self, volume: np.ndarray, axis: int) -> np.ndarray:
        feats_list = []
        batch = []
        for sl in _iter_slices(volume, axis=axis, stride=self.stride):
            batch.append(_slice_to_tensor(sl))
            if len(batch) == self.batch_size:
                x = torch.stack(batch).to(self.device)
                feats_list.append(self.extractor(x).cpu().numpy())
                batch = []
        if batch:
            x = torch.stack(batch).to(self.device)
            feats_list.append(self.extractor(x).cpu().numpy())
        if not feats_list:
            return np.zeros((0, _FEATURE_DIM), dtype=np.float32)
        return np.concatenate(feats_list, axis=0)

    def add_pair(self, real_vol: np.ndarray, fake_vol: np.ndarray) -> None:
        """Tek bir (real, fake) hacim çiftini işler. Çağrıdan sonra
        `real_vol`/`fake_vol` çağıran tarafta serbest bırakılabilir —
        burada hiçbir referans tutulmuyor, sadece küçük feature'lar."""
        for plane, axis in _AXIS_FOR_PLANE.items():
            self.real_moments[plane].add(self._extract_volume_features(real_vol, axis))
            self.fake_moments[plane].add(self._extract_volume_features(fake_vol, axis))

    def finalize(self) -> dict:
        results = {}
        for plane in _AXIS_FOR_PLANE:
            mu_r, sigma_r, n_r = self.real_moments[plane].finalize()
            mu_f, sigma_f, n_f = self.fake_moments[plane].finalize()
            if n_r < 2 or n_f < 2:
                results[f"FID_2p5D_{plane}"] = float("nan")
                continue
            results[f"FID_2p5D_{plane}"] = _frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

        valid_vals = [v for v in results.values() if not np.isnan(v)]
        results["FID_2p5D_Avg"] = float(np.mean(valid_vals)) if valid_vals else float("nan")
        return results


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


if __name__ == "__main__":
    print("Duman testi: streaming FID accumulator, rastgele hacimlerle...")
    rng = np.random.default_rng(0)
    acc = FIDAccumulator(device="cpu", stride=8)
    for _ in range(3):
        real = rng.normal(0.5, 0.15, size=(48, 48, 48)).astype(np.float32)
        fake = rng.normal(0.5, 0.15, size=(48, 48, 48)).astype(np.float32)
        acc.add_pair(real, fake)
        del real, fake  # gerçek kullanımda da böyle serbest bırakılacak

    out = acc.finalize()
    print(out)
    assert "FID_2p5D_Avg" in out
    print("Duman testi tamamlandı — hacimler tek tek işlendi, toplu tutulmadı.")
