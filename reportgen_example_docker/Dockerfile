FROM python:3.12

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app/gc_job/ /input /output \
    && chown user:user /opt/app/ /input /output

USER user
WORKDIR /opt/app/

ENV PATH="/home/user/.local/bin:${PATH}"
COPY --chown=user:user ct_chat /opt/app/ct_chat
COPY --chown=user:user models /opt/app/models
COPY --chown=user:user CT-CLIP /opt/app/CT-CLIP
COPY --chown=user:user llama /opt/app/llama


RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m piptools sync requirements.txt

RUN pip install -e ./ct_chat 
WORKDIR /opt/app/CT-CLIP
RUN pip install -e ./transformer_maskgit  && pip install -e ./CT_CLIP
WORKDIR /opt/app/




COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "process.py" ]
