# VDCT pod image: the setup_pod.sh `env` layer, baked.
#
# Build AFTER a bring-up has validated the layer -- the point of an image is to
# freeze a KNOWN-GOOD recipe, not to guess one. See docker/README.md.
#
# Build for the pod's architecture (linux/amd64), not the Mac's:
#   docker buildx build --platform linux/amd64 \
#       -t ghcr.io/<owner>/vdct-verl:<verl-sha> --push \
#       experiments/vdct_verl/docker
#
# The MODEL IS DELIBERATELY NOT BAKED IN (~16GB). It lives on the network
# volume at /workspace/models, which persists across pods, so baking it would
# inflate every pull to save a download we only do once per volume.
FROM verlai/verl:vllm024.dev2

ARG VERL_SHA=2b0fe51

# RunPod needs sshd; the verl image ships no openssh package.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server git \
    && mkdir -p /run/sshd \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Layer onto the image's python -- NOT a fresh venv, which would discard the
# pinned torch/vLLM/flash-attn closure this image exists to provide.
# transformers: verl main requires >=5.5.3 (setup.py:43); the image ARG pins 5.3.0.
# zstandard: slime_port.rollout_writer imports it.
RUN uv pip install --system --no-cache \
        "transformers>=5.5.3,!=5.6.0,<5.11" \
        zstandard \
        "huggingface_hub[cli]"

# verl is not baked into the base image (its Dockerfile installs it for the
# dependency closure, then pip-uninstalls). --no-deps keeps the pinned stack.
RUN git clone --filter=blob:none https://github.com/volcengine/verl /opt/verl \
    && git -C /opt/verl checkout -q "${VERL_SHA}" \
    && uv pip install --system --no-deps -e /opt/verl

ENV VERL_DIR=/opt/verl
WORKDIR /workspace
