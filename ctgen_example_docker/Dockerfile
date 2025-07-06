FROM python:3.12

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app/gc_job/ /input /output \
    && chown user:user /opt/app/ /input /output

USER user
WORKDIR /opt/app/

ENV PATH="/home/user/.local/bin:${PATH}"
COPY --chown=user:user GenerateCT /opt/app/GenerateCT
COPY --chown=user:user models /opt/app/models


RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m piptools sync requirements.txt

WORKDIR /opt/app/GenerateCT
RUN pip install -e ./transformer_maskgit  && pip install -e ./super_resolution
WORKDIR /opt/app/




COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "process.py" ]
