# syntax=docker/dockerfile:1.7
FROM python:3.12.12-slim-bookworm AS runtime
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.8.0 --index-url ${TORCH_INDEX}
WORKDIR /app
COPY pyproject.toml ./
COPY constraints.txt /opt/kev-constraints.txt
RUN git init /opt/kev && git -C /opt/kev remote add origin https://github.com/jaredpalmer/kev.git \
    && git -C /opt/kev fetch --depth 1 origin 65b1e763943ace53fd3a4aae00d8d08dbd849772 \
    && git -C /opt/kev checkout --detach FETCH_HEAD \
    && python -c "import subprocess,tomllib; subprocess.run(['pip','install','-c','/opt/kev-constraints.txt',*tomllib.load(open('pyproject.toml','rb'))['project']['dependencies'],'/opt/kev[serve]'],check=True)" \
    && pip check \
    && rm -rf /opt/kev/.git \
    && pip freeze > /opt/installed-requirements.txt
COPY kev_service ./kev_service
RUN pip install --no-deps . && pip freeze > /opt/installed-requirements.txt

FROM runtime AS model-tools
ENTRYPOINT ["python", "-m", "kev_service.download"]

FROM runtime AS test
COPY tests ./tests
COPY examples ./examples
RUN pip install '.[test]' && python -m pytest -q

FROM runtime AS production
ARG KEV_BUNDLE_MODEL=kev-0.8b
# The cache can hold multiple models. Only this directory is copied into the output layer.
RUN --mount=type=bind,source=model-cache,target=/model-cache \
    python -m kev_service.download --archive /model-cache/${KEV_BUNDLE_MODEL}.tar --copy-to /opt/models
RUN groupadd --gid 10001 kev && useradd --uid 10001 --gid 10001 --no-create-home kev \
    && mkdir /data /debug && chown kev:kev /data /debug \
    && find /opt/models -type d ! -perm -005 -exec chmod a+rx {} + \
    && find /opt/models -type f ! -perm -004 -exec chmod a+r {} +
ENV KEV_MODEL=${KEV_BUNDLE_MODEL} KEV_DEVICE=cpu KEV_DTYPE=auto \
    HF_HOME=/opt/models/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
USER 10001:10001
EXPOSE 8009
HEALTHCHECK --interval=15s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('KEV_PORT','8009')+'/health/ready',timeout=4)"
ENTRYPOINT ["python", "-m", "kev_service"]

