# Container image for BOTH monitor Lambdas (poller + alerter).
# One image, two functions — each function overrides CMD with its handler.
#
# Build context = repo root:
#   docker build -f monitor/Dockerfile -t indot-bridge-monitor .
#
# micromamba base so eccodes/cfgrib (GRIB) install cleanly from conda-forge;
# awslambdaric is the AWS Lambda Runtime Interface Client that makes an
# arbitrary container work as a Lambda.
FROM mambaorg/micromamba:1.5.8

COPY --chown=$MAMBA_USER:$MAMBA_USER monitor/env-lambda.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean --all --yes

# Activate the env for subsequent RUN/ENTRYPOINT (micromamba convention).
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --no-cache-dir awslambdaric

WORKDIR /var/task
COPY --chown=$MAMBA_USER:$MAMBA_USER monitor/monitor_common ./monitor_common
COPY --chown=$MAMBA_USER:$MAMBA_USER monitor/lambda_poller.py monitor/lambda_alerter.py ./
ENV PYTHONPATH=/var/task

# _entrypoint.sh activates the conda env, then runs the Lambda RIC.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "-m", "awslambdaric"]
CMD ["lambda_poller.handler"]
