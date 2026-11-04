# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
ARG CPU_BASE_IMAGE=python:3.12-slim-bookworm
ARG A2_BASE_IMAGE=ascendai/cann:9.2.0-910b-ubuntu22.04-py3.12
ARG ASCEND_REPO=https://gitcode.com/Ascend/torchtitan-ascend.git
ARG ASCEND_BRANCH=refactor/npu-override

FROM ${CPU_BASE_IMAGE} AS cpu
ARG MAX_JOBS=32
# Corporate TLS-inspecting gateway CA (PEM). Empty by default so the image stays
# buildable outside the corporate network.
ARG CORPORATE_CA=
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:${PATH}
ENV UV_CONSTRAINT=/opt/torchtitan-ascend/constraints.txt \
    UV_OVERRIDE=/opt/torchtitan-ascend/overrides.txt \
    UV_SYSTEM_CERTS=1 \
    CI_DEVICE=cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libnuma1 \
        ca-certificates curl wget vim less jq procps \
        iproute2 iputils-ping netcat-openbsd rsync unzip \
    && rm -rf /var/lib/apt/lists/* \
    && printf '%s\n' "alias ll='ls -alF'" >> /etc/bash.bashrc \
    && python -m venv /opt/venv \
    && pip install --no-cache-dir uv
# Trust the inspecting gateway. uv and pip validate against certifi when
# available, so append the PEM there as well as to the system bundle.
RUN if [ -n "$CORPORATE_CA" ]; then \
        printf '%s\n' "$CORPORATE_CA" > /usr/local/share/ca-certificates/corporate-gateway.crt && \
        cat /usr/local/share/ca-certificates/corporate-gateway.crt >> /etc/ssl/certs/ca-certificates.crt && \
        update-ca-certificates && \
        python -c "import certifi, pathlib; p = pathlib.Path(certifi.where()); p.write_text(p.read_text() + '\n' + open('/usr/local/share/ca-certificates/corporate-gateway.crt').read())" 2>/dev/null || true; \
    fi
ARG ASCEND_REPO
ARG ASCEND_BRANCH
RUN git clone --depth=1 --branch "$ASCEND_BRANCH" \
    "$ASCEND_REPO" /opt/torchtitan-ascend
RUN bash /opt/torchtitan-ascend/ci/install.sh cpu
WORKDIR /opt/torchtitan
ENTRYPOINT ["bash", "/opt/torchtitan-ascend/ci/entrypoint.sh"]
CMD ["bash"]

FROM ${A2_BASE_IMAGE} AS a2
ARG MAX_JOBS=32
# See the cpu stage: same corporate gateway CA handling.
ARG CORPORATE_CA=
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/usr/lib/llvm-15/bin:${PATH}
ENV UV_CONSTRAINT=/opt/torchtitan-ascend/constraints.txt \
    UV_OVERRIDE=/opt/torchtitan-ascend/overrides.txt \
    UV_SYSTEM_CERTS=1 \
    CI_DEVICE=a2 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=1 \
    TORCH_NPU_DEVICE_CAPABILITY=8.0
# gawk is required by the fla_npu AscendC build scripts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential gawk clang-15 lld-15 zlib1g-dev libzstd-dev libnuma-dev \
        ca-certificates curl wget vim less jq procps \
        iproute2 iputils-ping netcat-openbsd rsync unzip \
    && rm -rf /var/lib/apt/lists/* \
    && printf '%s\n' "alias ll='ls -alF'" >> /etc/bash.bashrc \
    && python -m venv /opt/venv \
    && pip install --no-cache-dir uv
RUN if [ -n "$CORPORATE_CA" ]; then \
        printf '%s\n' "$CORPORATE_CA" > /usr/local/share/ca-certificates/corporate-gateway.crt && \
        cat /usr/local/share/ca-certificates/corporate-gateway.crt >> /etc/ssl/certs/ca-certificates.crt && \
        update-ca-certificates && \
        python -c "import certifi, pathlib; p = pathlib.Path(certifi.where()); p.write_text(p.read_text() + '\n' + open('/usr/local/share/ca-certificates/corporate-gateway.crt').read())" 2>/dev/null || true; \
    fi
ARG ASCEND_REPO
ARG ASCEND_BRANCH
RUN git clone --depth=1 --branch "$ASCEND_BRANCH" \
    "$ASCEND_REPO" /opt/torchtitan-ascend
RUN bash /opt/torchtitan-ascend/ci/install.sh a2
WORKDIR /opt/torchtitan
ENTRYPOINT ["bash", "/opt/torchtitan-ascend/ci/entrypoint.sh"]
CMD ["bash"]
