from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import numpy as np
import torch

from FVD import fvd_pytorch_model
from FVD.util import open_url

@torch.no_grad()
def compute_fvd(videos_fake: np.ndarray, videos_real: np.ndarray, device: str='cuda') -> float:
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.

    with open_url(detector_url, verbose=False) as f:
        detector = torch.jit.load(f).eval().to(device)

    videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
    videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)

    feats_fake = detector(videos_fake, **detector_kwargs).cpu().numpy()
    feats_real = detector(videos_real, **detector_kwargs).cpu().numpy()

    return fvd_pytorch_model.compute_fvd(feats_fake, feats_real)