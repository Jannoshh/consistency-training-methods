# VDCT pod image

Bakes the `scripts/setup_pod.sh env` layer on top of `verlai/verl:vllm024.dev2`
so a fresh pod skips the apt/pip/verl-install step.

## Sequencing: script first, image second

`setup_pod.sh` is the source of truth and the Dockerfile mirrors it. Build the
image only from a layer a real bring-up has already validated — freezing a
guessed recipe into an image buys nothing and hides the failure one level down.

## What it is and isn't worth

Baked: apt (`openssh-server`), the transformers pin, `zstandard`, `uv`, and the
verl clone installed `--no-deps`. That is the part that does NOT survive a pod
teardown, because it lives in the container's system python rather than on the
volume.

NOT baked: the **model** (~16GB) and the **data**. Both live on the network
volume at `/workspace`, which persists across pods — baking them would inflate
every pull to save a one-off download. Keep the volume and a new pod already has
them.

So the image saves the apt+pip minutes, and the volume saves the model download.
They are complementary, and the volume is the bigger win.

## Building

Do NOT build on an arm64 Mac without `--platform linux/amd64`; the pod is amd64
and an arm64 image will not run there.

```bash
docker buildx build --platform linux/amd64 \
    -t ghcr.io/<owner>/vdct-verl:2b0fe51 --push \
    experiments/vdct_verl/docker
```

Tag with the verl SHA — the pinned SHA is the thing that makes the layer
reproducible, so it belongs in the tag rather than `latest`.

Then pass `--imageName ghcr.io/<owner>/vdct-verl:2b0fe51` to `runpodctl create
pod`, and a private registry needs credentials registered with RunPod first.
