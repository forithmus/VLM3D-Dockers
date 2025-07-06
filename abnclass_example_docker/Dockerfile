FROM python:3.11.5

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app/gc_job/ /input /output \
    && chown user:user /opt/app/ /input /output

USER user
WORKDIR /opt/app/

ENV PATH="/home/user/.local/bin:${PATH}"
COPY --chown=user:user models/CT-CLIP-Related/ /opt/app/models
COPY --chown=user:user CT-CLIP /opt/app/CT-CLIP
COPY --chown=user:user BiomedVLP-CXR-BERT-specialized/ /opt/app/models/BiomedVLP-CXR-BERT-specialized


RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m piptools sync requirements.txt

WORKDIR /opt/app/CT-CLIP
RUN pip install -e ./transformer_maskgit  && pip install -e ./CT_CLIP
WORKDIR /opt/app/




COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "process.py" ]
