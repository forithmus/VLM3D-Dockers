# MR volume-generation evaluation container (self-contained).
# Scores generated .nii.gz volumes against references: MSE / PSNR / SSIM
# per modality plus 2.5D FID over squeezenet1_1 features.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir nibabel scipy scikit-image torchvision

# Offline at runtime: bake the squeezenet weights into the image at build time
RUN python -c "import torchvision.models as m; m.squeezenet1_1(weights=m.SqueezeNet1_1_Weights.IMAGENET1K_V1)"

COPY score.py fid_2p5d.py metrics_basic.py modality_filter.py /app/
WORKDIR /app
ENTRYPOINT ["python3", "/app/score.py"]
