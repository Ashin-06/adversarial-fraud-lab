# Runtime image: dependencies from the pinned lock, source copied in, runs as a non-root user.
FROM python:3.12-slim-bookworm AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.lock /tmp/
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r /tmp/requirements.lock

FROM python:3.12-slim-bookworm
# the app user has no home directory; give matplotlib a writable cache
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    ADVLAB_ENV=production ADVLAB_HOST=0.0.0.0 ADVLAB_PORT=8005 \
    ADVLAB_DATABASE_URL=sqlite:////data/advlab.db ADVLAB_MODEL_PATH=/models/lab.joblib
RUN useradd --system --uid 10001 --no-create-home app && mkdir /data && chown app /data
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY advlab advlab
USER app
VOLUME /data
EXPOSE 8005
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ['ADVLAB_PORT'], timeout=2)"]
CMD ["python", "-m", "advlab", "serve"]
