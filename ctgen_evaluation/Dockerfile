FROM python:3.11.5

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p /opt/app /input /output && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"


USER root
RUN apt-get update && apt-get install -y default-jre && rm -rf /var/lib/apt/lists/*
USER user

COPY --chown=user:user \
    evaluation.py evaluate_fvd.py evaluate_clip.py evaluate_cpu.py evaluate_fid_2p5d_ct.py \
    metrics.py metrics3d.py compute_fid_2-5d_ct.py data_input.xlsx ResNet50.pt \
    ./

# Hidden GT and test material (if required at runtime)
COPY --chown=user:user ground-truth/ ground-truth/
COPY --chown=user:user test/          test/
COPY --chown=user:user CT-CLIP/ /opt/app/CT-CLIP/
COPY --chown=user:user FVD/ /opt/app/FVD/
COPY --chown=user:user BiomedVLP-CXR-BERT-specialized/ /opt/app/models/BiomedVLP-CXR-BERT-specialized
COPY --chown=user:user weights/models/CT-CLIP-Related/ /opt/app/models


RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
COPY --chown=user:user requirements.txt .
RUN pip install monai
WORKDIR /opt/app/CT-CLIP
RUN pip install -e ./transformer_maskgit  && pip install -e ./CT_CLIP
WORKDIR /opt/app/FVD
RUN pip install -e .
WORKDIR /opt/app/
RUN pip install -r requirements.txt          # or 

RUN python3 -c "import torch; torch.hub.load('Warvito/radimagenet-models', model='radimagenet_resnet50', trust_repo=True)"

VOLUME /opt/app/test        # optional – keeps host test dir writable

ENTRYPOINT ["python", "-m", "evaluation"]
