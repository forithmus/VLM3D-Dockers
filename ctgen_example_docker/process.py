#!/usr/bin/env python3
"""
generate_ct_pipeline.py
=======================

*No runtime flags required.*
Generates 3D CT scans from text prompts in two stages:
1. Transformer-based low-resolution generation.
2. Diffusion-based super-resolution.

The script assumes a fixed container layout:

/input/prompts.json   ← Input JSON with text prompts
/output/              ← Writable directory for the resulting *.mha files

/opt/app/models/ctvit_pretrained.pt        ← Pre-trained CTViT weights
/opt/app/models/transformer_pretrained.pt  ← Pre-trained Transformer weights
/opt/app/models/superres_pretrained.pt     ← Pre-trained Super-Resolution weights
"""
from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from accelerate import Accelerator
from einops import rearrange
from tqdm import tqdm

# --- Assume these are importable from your project structure ---
try:
    from transformer_maskgit import CTViT, MaskGit, MaskGITTransformer
    from super_resolution import Unet, ElucidatedSuperres, Superres, SuperResolutionTrainer, NullUnet
except ImportError as e:
    print(f"✗ Failed to import custom modules: {e}")
    print("✗ Please ensure transformer_maskgit and super_resolution modules are in the PYTHONPATH.")
    sys.exit(1)

# ---------------------------------------------------------------------------
#  HARD-WIRED PATHS & PARAMETERS
# ---------------------------------------------------------------------------
INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")
MODELS_DIR = Path("/opt/app/models")

# Model Paths
CTVIT_WEIGHTS_PATH = MODELS_DIR / "ctvit_pretrained.pt"
TRANSFORMER_WEIGHTS_PATH = MODELS_DIR / "transformer_pretrained.pt"
SUPERRES_WEIGHTS_PATH = MODELS_DIR / "superres_pretrained.pt"

# Generation Parameters (derived from configs)
TRANSFORMER_NUM_FRAMES = 201
TRANSFORMER_COND_SCALE = 5.0
LOW_RES_IMG_SIZE = 128
HIGH_RES_IMG_SIZE = 512       # From superres.params.image_sizes
SUPERRES_COND_SCALE = 1.0     # From checkpoint.cond_scale

# Final Output Parameters
FINAL_SPACING = (1.0, 1.0, 1.0)  # (x, y, z) mm
VALUE_RANGE = (-1000, 1000)

# ---------------------------------------------------------------------------
#  HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def tensor_to_nifti(tensor, filename):
    """Saves a tensor to a NIfTI file."""
    tensor = tensor.permute(3, 2, 1, 0).cpu().numpy()
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(tensor, affine)
    nib.save(nifti_img, filename)


def setup_transformer_model():
    """Loads and prepares the Stage 1 Transformer model."""
    print("Setting up Stage 1: Transformer Model...")
    ctvit = CTViT(
        dim=512, codebook_size=8192, image_size=LOW_RES_IMG_SIZE, patch_size=16,
        temporal_patch_size=2, spatial_depth=4, temporal_depth=4, dim_head=32, heads=8
    )
    ctvit.load(str(CTVIT_WEIGHTS_PATH))
    ctvit.eval()

    maskgit = MaskGit(
        num_tokens=8192, max_seq_len=10000, dim=512, dim_context=768, depth=6
    )

    transformer_model = MaskGITTransformer(ctvit=ctvit, maskgit=maskgit).cuda()
    transformer_model.load(str(TRANSFORMER_WEIGHTS_PATH))
    transformer_model.eval()
    print("✓ Transformer model loaded.")
    return transformer_model


def setup_superres_model():
    """
    Loads and prepares the Stage 2 Super-Resolution model.
    Parameters are strictly derived from the provided superres_inference.yaml.
    """
    print("Setting up Stage 2: Super-Resolution Model...")

    # The first unet in the cascade is a NullUnet (no-op)
    unet1 = NullUnet()

    # The second unet parameters are from the YAML config's 'unets.unet1' section
    unet2 = Unet(
        dim=64,
        num_resnet_blocks=2,
        dim_mults=[1, 2, 4],
        layer_cross_attns=[False, False, True],
        use_linear_attn=False,
        layer_attns=False,
        attend_at_middle=False,
        use_linear_cross_attn=False,
        memory_efficient=True,
        channels=1,
        attn_heads=8,
    )

    superres = Superres(
        unets=(unet1, unet2),
        image_sizes=[LOW_RES_IMG_SIZE, HIGH_RES_IMG_SIZE],
        channels=1,
        timesteps=[25, 25],
        condition_on_text=True,
        random_crop_sizes=[None, None],
    )

    trainer = SuperResolutionTrainer(
        superres=superres,
        use_ema=False
    ).cuda()

    trainer.load(str(SUPERRES_WEIGHTS_PATH))
    trainer.eval()
    print("✓ Super-Resolution model loaded.")
    return trainer


def save_final_mha(tensor: torch.Tensor, output_path: Path):
    """Clips, converts, and saves the final tensor as an MHA file with correct metadata."""
    if tensor.shape[0] != 1:
        raise ValueError("Expected single channel tensor.")

    array = tensor.squeeze(0).cpu().numpy()
    array = np.clip(array, 0, 1)
    array = array * 2000
    array = array - 1000
    array = np.clip(array, VALUE_RANGE[0], VALUE_RANGE[1])

    itk_image = sitk.GetImageFromArray(array)
    itk_image.SetSpacing(FINAL_SPACING)
    itk_image.SetOrigin((0, 0, 0))

    sitk.WriteImage(itk_image, str(output_path))


# ---------------------------------------------------------------------------
#  MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    try:
        input_json_path = next(INPUT_DIR.glob("*.json"))
        print(f"Found input JSON: {input_json_path}")
    except StopIteration:
        print(f"✗ No JSON file found in {INPUT_DIR}")
        sys.exit(1)

    with input_json_path.open("r") as f:
        prompts_data = json.load(f)

    if not isinstance(prompts_data, list):
        print("✗ Input JSON must be a list of objects.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transformer_model = setup_transformer_model()
    superres_trainer = setup_superres_model()
    accelerator = superres_trainer.accelerator
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        print(f"Using temporary directory for intermediate files: {tmp_path}")

        for item in tqdm(prompts_data, desc="Generating CTs"):
            prompt = item["report"]
            output_filename = Path(item["input_image_name"]).stem
            tqdm.write(f"\nProcessing: {output_filename}")

            tqdm.write("  → Stage 1: Generating low-resolution volume...")
            with torch.no_grad():
                low_res_tensor = transformer_model.sample(
                    texts=[prompt],
                    num_frames=TRANSFORMER_NUM_FRAMES,
                    cond_scale=TRANSFORMER_COND_SCALE
                )

            low_res_nifti_path = tmp_path / f"{output_filename}_lowres.nii.gz"
            tensor_to_nifti(low_res_tensor.squeeze(0), low_res_nifti_path)
            tqdm.write(f"  ✓ Saved low-res NIfTI to {low_res_nifti_path}")

            tqdm.write("  → Stage 2: Applying super-resolution...")
            low_res_slices = low_res_tensor.permute(0, 2, 1, 3, 4).squeeze(0)
            low_res_slices = np.clip(low_res_slices.cpu().numpy(), -1, 1)
            low_res_slices = (low_res_slices + 1) / 2
            low_res_slices = torch.tensor(low_res_slices).cuda()
            high_res_slices = []
            with torch.no_grad():
                for slice_tensor in tqdm(low_res_slices, desc="  Super-resolving slices", leave=False):
                    input_slice = slice_tensor.unsqueeze(0).cuda()

                    high_res_slice = superres_trainer.sample(
                        start_image_or_video=input_slice,
                        texts=[prompt],
                        cond_scale=SUPERRES_COND_SCALE,
                        start_at_unet_number=2,
                    ).detach().cpu()
                    high_res_slices.append(high_res_slice.squeeze(0))

            high_res_volume = torch.stack(high_res_slices, dim=0)
            high_res_volume = high_res_volume.permute(1, 0, 3, 2)
            tqdm.write("  ✓ Super-resolution complete.")

            tqdm.write("  → Stage 3: Finalizing and saving to MHA...")
            final_output_path = OUTPUT_DIR / f"{output_filename}.mha"
            save_final_mha(high_res_volume, final_output_path)
            tqdm.write(f"  ✓ Saved final output to {final_output_path}")

    print("\n✅ Pipeline finished successfully.")

    # -----------------------------------------------------------------------
    #  ZIP & CLEANUP
    # -----------------------------------------------------------------------
    print("📦 Zipping all .mha files into outputs.zip…")
    zip_path = OUTPUT_DIR / "predictions.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for mha_file in OUTPUT_DIR.glob("*.mha"):
            zipf.write(mha_file, arcname=mha_file.name)
            mha_file.unlink()
    print(f"📦 Created archive at {zip_path} and removed original .mha files.")


if __name__ == "__main__":
    main()
