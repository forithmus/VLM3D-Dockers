# metrics3d.py
import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from FVD import fvd_pytorch as fvd        # pip install fvd_pytorch
# (you already used this)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- helpers ----------------------------------------------------------

def _to_uint8_3ch(x: torch.Tensor) -> torch.Tensor:
    """
    x : (B=1, 1, H, W) float in [0,1]  ->  (B=1, 3, H, W) uint8 in [0,255]
    """
    if x.dim() == 3:          # (H,W)  → (1,1,H,W)
        x = x.unsqueeze(1)
    if x.size(1) == 1:        # replicate gray → rgb
        x = x.repeat_interleave(3, dim=1)
    print(x.shape, "shape 2")
    print(x.min())
    print(x.max())
    x = (x.clamp(0, 1) * 255).to(torch.uint8)
    return x


def _resize_vol(vol: torch.Tensor,
                target_dhw=(201, 224, 224)) -> torch.Tensor:
    """
    vol : (D, H, W) → (1, 1, D, H, W) resized to target
    """
    d, h, w = target_dhw
    vol = vol.unsqueeze(0).unsqueeze(0)                      # (1,1,D,H,W)
    vol = F.interpolate(vol, size=(d, h, w),
                        mode="trilinear", align_corners=False)
    return vol.squeeze(0)            # (1, D, H, W)


# ---------- metrics ----------------------------------------------------------

@torch.no_grad()
def fid_volume(gen: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    gen, gt :  (D, H, W)  float in [0,1]
    Returns
    -------
    scalar tensor (Frechet Inception Distance for that one pair)
    """
    D = int(gen.shape[0])
    fid = FrechetInceptionDistance(feature=2048).to(DEVICE)

    for z in range(D):
        g_slice = _to_uint8_3ch(gen[z]).to(DEVICE)
        r_slice = _to_uint8_3ch(gt[z]).to(DEVICE)

        fid.update(r_slice, real=True)
        fid.update(g_slice, real=False)

    return fid.compute()            # tensor on DEVICE


@torch.no_grad()
def fvd_pair(gen: torch.Tensor,
             gt: torch.Tensor,
             target_dhw=(201, 224, 224)) -> float:
    """
    Compute FVD for one generated/GT pair.

    gen, gt : (D, H, W) float in [-1,1]  (FVD expects [-1,1])
    """
    print(gen.shape)
    print(gt.shape)
    gen_r = _resize_vol(gen, target_dhw)   # (1,D,H,W)
    gt_r  = _resize_vol(gt,  target_dhw)

    # add channel dim, then move dims → (T, H, W, C)
    gen_r = gen_r.repeat_interleave(3, dim=0).permute(1, 2, 3, 0)  # (D,H,W,3)
    gt_r  = gt_r.repeat_interleave(3, dim=0).permute(1, 2, 3, 0)

    # fvd_pytorch expects NumPy float32 in [-1,1], shape (N, T, H, W, C)
    return fvd.compute_fvd(
        gt_r.unsqueeze(0).cpu().numpy().astype(np.float32),
        gen_r.unsqueeze(0).cpu().numpy().astype(np.float32)
    )
