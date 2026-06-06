"""
Copy-pasted from https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""
from typing import Tuple
import scipy
import numpy as np


def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    # With CHUNK=4 samples and d=96 features, sigma is rank-deficient
    # → sigma_gen @ sigma_real has near-zero negative eigenvalues from
    # floating-point noise → scipy.linalg.sqrtm returns NaN. Regularise
    # with eps*I so the product stays PSD.
    eps = 1e-6 * np.trace(sigma_gen + sigma_real) / sigma_gen.shape[0]
    I = np.eye(sigma_gen.shape[0], dtype=sigma_gen.dtype)
    sigma_gen = sigma_gen + eps * I
    sigma_real = sigma_real + eps * I

    m = np.square(mu_gen - mu_real).sum()
    test = np.dot(sigma_gen, sigma_real)
    s, _ = scipy.linalg.sqrtm(test, disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    print(feats.shape)
    sigma = np.cov(feats, rowvar=False) # [d, d]

    return mu, sigma