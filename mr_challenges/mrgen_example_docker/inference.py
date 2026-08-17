"""
MR Volume Generation - Inference Script
"""
import json
from pathlib import Path
import argparse
import torch
import numpy as np
import nibabel as nib
import os
import shutil
import signal
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoModel
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

WEIGHTS_DIR = "/tmp/forithmus/weights-extracted"

def load_models(device):
    print(f"[TESHIS] torch surumu: {torch.__version__}")
    print(f"[TESHIS] torch CUDA surumu: {torch.version.cuda}")
    print(f"[TESHIS] CUDA mevcut mu: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[TESHIS] GPU adi: {torch.cuda.get_device_name(0)}")
    else:
        print("[TESHIS] CUDA mevcut degil")
    print(f"[TESHIS] {WEIGHTS_DIR} icerigi:")
    for root, dirs, files in os.walk(WEIGHTS_DIR):
        for f in files:
            print(f"[TESHIS]   {os.path.join(root, f)}")

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True
    )
    text_encoder = AutoModel.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True
    ).to(device)
    text_encoder.eval()

    autoencoder = AutoencoderKlMaisi(
        spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4,
        num_channels=[64, 128, 256], num_res_blocks=[2, 2, 2],
        norm_num_groups=32, norm_eps=1e-6,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
        num_splits=8, dim_split=1,
    )
    ae_ckpt = torch.load(f"{WEIGHTS_DIR}/autoencoder_v1.pt", map_location="cpu")
    autoencoder.load_state_dict(ae_ckpt)
    autoencoder.to(device)
    autoencoder.eval()

    unet = DiffusionModelUNetMaisi(
        spatial_dims=3, in_channels=4, out_channels=4,
        num_channels=[64, 128, 256, 512],
        attention_levels=[False, False, True, True],
        num_head_channels=[0, 0, 32, 32],
        num_res_blocks=2, use_flash_attention=torch.cuda.is_available(),
        include_spacing_input=True, num_class_embeds=128,
        resblock_updown=True, include_fc=True,
        with_conditioning=True, cross_attention_dim=768,
    )
    ft_ckpt = torch.load(
        f"{WEIGHTS_DIR}/finetuned_text_conditioned_unet.pt",
        map_location="cpu", weights_only=False,
    )
    unet.load_state_dict(ft_ckpt["model_state_dict"])
    unet.to(device)
    unet.eval()
    unet = unet.to(torch.bfloat16)

    # Bu checkpoint'te scale_factor kaydedilmemiÅŸ â€” proje boyunca
    # kullanÄ±lan sabit deÄŸer (RFlowScheduler kurulumunda hesaplanan).
    scale_factor = 0.9704500436782837

    return tokenizer, text_encoder, autoencoder, unet, scale_factor


def detect_modality(report_text):
    """Rapor metninden hangi MRI sekansÄ±nÄ±n istendigini tahmin eder."""
    text = report_text.lower()
    if "flair" in text:
        return 11
    if "t2" in text:
        return 10
    if "swi" in text:
        return 20
    return 9  # varsayilan: T1-weighted


def generate_batch(report_texts, tokenizer, text_encoder, autoencoder, unet, scale_factor, device,
                    num_inference_steps=30, spacing=(1.5, 1.9, 1.9)):
    B = len(report_texts)
    tokens = tokenizer(report_texts, return_tensors="pt", truncation=True,
                        max_length=256, padding=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        text_emb = text_encoder(**tokens).last_hidden_state
        text_emb = text_emb.to(torch.bfloat16)

    modality_ids = [detect_modality(t) for t in report_texts]
    modality = torch.tensor(modality_ids).to(device)
    spacing_tensor = torch.tensor([list(spacing)] * B).to(device)
    spacing_tensor = spacing_tensor.to(torch.bfloat16)

    x = torch.randn(B, 4, 64, 64, 64).to(device)
    dt = 1.0 / num_inference_steps

    with torch.no_grad():
        for step in range(num_inference_steps):
            t_frac = 1.0 - step / num_inference_steps
            t_raw = torch.full((B,), t_frac * 1000.0).to(device)
            t_raw = t_raw.to(torch.bfloat16)
            x_bf16 = x.to(torch.bfloat16)
            velocity = unet(
                x=x_bf16, timesteps=t_raw, context=text_emb,
                class_labels=modality, spacing_tensor=spacing_tensor,
            )
            velocity = velocity.to(torch.float32)
            x = x + dt * velocity

        # Decode'u tek tek yap (decoder bellek acidir, batch halinde OOM veriyor)
        latents = (x / scale_factor).float()
        decoded_list = []
        for b in range(B):
            single = autoencoder.decode_stage_2_outputs(latents[b:b+1])
            decoded_list.append(single.cpu().numpy())
        # empty_cache() satırını kaldırdık

    return np.concatenate(decoded_list, axis=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    input_dir = Path("/input")
    output_dir = Path("/output")
    checkpoint_dir = Path("/checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / "progress.json"
    output_backup = checkpoint_dir / "outputs"
    output_backup.mkdir(parents=True, exist_ok=True)

    processed = []
    shutting_down = {"flag": False}

    def save_checkpoint():
        with open(checkpoint_file, "w") as f:
            json.dump({"processed": processed}, f)
        for fname in processed:
            src = output_dir / fname
            dst = output_backup / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    def handle_sigterm(sig, frame):
        shutting_down["flag"] = True
        print("SIGTERM alindi, checkpoint kaydediliyor...")
        save_checkpoint()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    prompts_path = input_dir / "prompts.json"
    if not prompts_path.exists():
        candidates = list(input_dir.glob("*.json"))
        prompts_path = candidates[0]

    with open(prompts_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    print(f"{len(entries)} rapor bulundu.")

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            processed = json.load(f)["processed"]
        print(f"Devam ediliyor: {len(processed)} kayit zaten islenmis.")
        for fname in processed:
            bak = output_backup / fname
            out = output_dir / fname
            if bak.exists():
                shutil.copy2(bak, out)

    tokenizer, text_encoder, autoencoder, unet, scale_factor = load_models(device)
    print("Modeller yuklendi.")

    BATCH_SIZE = 16  # A100 40GB için başlangıç değeri, gerekirse düşür/yükselt

    processed_set = set(processed)
    pending = [e for e in entries if f"{e['input_image_name']}.nii.gz" not in processed_set]

    for batch_start in range(0, len(pending), BATCH_SIZE):
        if shutting_down["flag"]:
            break

        batch = pending[batch_start:batch_start + BATCH_SIZE]
        fnames = [f"{e['input_image_name']}.nii.gz" for e in batch]
        report_texts = [e["report"] for e in batch]

        volumes = generate_batch(report_texts, tokenizer, text_encoder, autoencoder, unet, scale_factor, device)

        for j, fname in enumerate(fnames):
            vol = volumes[j].squeeze()
            out_path = output_dir / fname
            nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), str(out_path))
            processed.append(fname)

        if len(processed) % (BATCH_SIZE * 1) == 0 or batch_start + BATCH_SIZE >= len(pending):
            save_checkpoint()
            print(f"Ilerleme: {len(processed)}/{len(entries)}")


if __name__ == "__main__":
    main()