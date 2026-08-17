"""
Temel metrikler: MSE, PSNR, SSIM — 3D hacim çiftleri için.
Aşama 1 (Blinded Classifier Consistency implemente edilmediği için
sadece bu metrikler + FID_2p5D kullanılıyor — bkz. score.py başındaki not).
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _normalize01(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    if hi - lo < 1e-6:
        return np.zeros_like(vol)
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0)


def compute_basic_metrics(real: np.ndarray, fake: np.ndarray) -> dict:
    """Tek bir (real, fake) hacim çifti için MSE/PSNR/SSIM hesaplar.
    İkisi de aynı shape'e sahip olmalı; değilse fake, real'in shape'ine
    en-yakın-komşu ile yeniden örneklenir (basit fallback)."""
    if real.shape != fake.shape:
        from scipy.ndimage import zoom

        factors = [r / f for r, f in zip(real.shape, fake.shape)]
        fake = zoom(fake, factors, order=1)

    real_n = _normalize01(real)
    fake_n = _normalize01(fake)

    mse = float(np.mean((real_n - fake_n) ** 2))
    psnr = float(peak_signal_noise_ratio(real_n, fake_n, data_range=1.0))
    ssim = float(structural_similarity(real_n, fake_n, data_range=1.0))

    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.15, size=(48, 48, 48)).astype(np.float32)
    b = a + rng.normal(0, 0.05, size=a.shape).astype(np.float32)
    out = compute_basic_metrics(a, b)
    print(out)
    assert out["SSIM"] > 0.5  # yakın hacimler için yüksek SSIM bekleniyor
    print("Duman testi tamamlandı.")
