FROM python:3.10

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p /opt/app /input /output && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

USER root
RUN apt-get update && apt-get install -y default-jre && rm -rf /var/lib/apt/lists/*
USER user


COPY --chown=user:user requirements.txt .
RUN python -m piptools sync requirements.txt          # or  pip install -r requirements.txt

# ------------------------------------------------------------------ #
# Copy everything the evaluation pipeline needs
# ------------------------------------------------------------------ #
COPY --chown=user:user \
    evaluation.py calc_scores.py \
    crg_score.py json_to_csv.py \
    ./

# Hidden GT and test material (if required at runtime)
COPY --chown=user:user ground-truth/ ground-truth/
COPY --chown=user:user test/          test/


VOLUME /opt/app/test        # optional – keeps host test dir writable

ENTRYPOINT ["python", "-m", "evaluation"]
