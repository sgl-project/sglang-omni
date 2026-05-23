# 🚀 Get Started

## 📦 Installation

We highly recommend to use our [Docker Image](#🐳-use-docker) for development or production environment. Otherwise, please make sure you have built and installed [`ucx`](https://github.com/openucx/ucx) in your environment.

```bash
# clone this repository
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

# create a virtual environment in docker
uv venv .venv -p 3.12
source .venv/bin/activate

# install
uv pip install -v .

# install for development
uv pip install -v -e .
```

> **Note:** On Debian Bookworm / Ubuntu 23.04+ system Python (including the `frankleeeee/sglang-omni:dev` Docker image), use the `uv venv` flow above. Installing into system Python directly (e.g., `pip install -e .`) triggers [PEP 668](https://peps.python.org/pep-0668/):
>
> ```
> error: externally-managed-environment
> × This environment is externally managed
> ```
>
> Either use the venv flow, or pass `--break-system-packages` to the installer to override.


## 🐳 Use Docker

We have build all necessary dependencies into our Docker Image, so you can simply pull and run it.

```bash
# we strongly recommend using our docker image for stable environment
# NOTE: this docker image will be moved to lmsysorg upon release
docker pull frankleeeee/sglang-omni:dev

# run the container
docker run -it \
    --shm-size 32g \
    --gpus all \
    --ipc host \
    --network host \
    --privileged \
    frankleeeee/sglang-omni:dev \
    /bin/zsh
```
