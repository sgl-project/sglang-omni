# 🚀 Get Started

## 📦 Installation

We highly recommend to use our [Docker Image](#🐳-use-docker) for development or production environment. Otherwise, please make sure you have built and installed [`ucx`](https://github.com/openucx/ucx) in your environment.

```bash
# clone this repository
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

# create a virtual environment in docker
uv venv .venv -p 3.11
source .venv/bin/activate

# install
uv pip install -v .

# install for development
uv pip install -v -e ".[dev]"
```


## 🐳 Use Docker


### Option 1 Use Devcontainer (VSCode as an example)

[Devcontainer](https://containers.dev/) can auto spin up a development environment on top of docker image and handles host-container folder binding.

We will use VS Code as example here, but devbocontainer [supports](https://containers.dev/supporting) multiple ide and tools

See also: [sglang dev container guide](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/development_guide_using_docker.md#option-1-use-the-default-dev-container-automatically-from-vscode)

### Prerequisites

- VS Code installation on development host machine
  - You can find VS Code installation guide at https://code.visualstudio.com/download


### Use The Default Dev Container Automatically From VS Code

```bash
# clone this repository
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

# open the repository in VS Code locally
code .
# if you are opening a folder in remote host, use ssh:
code --remote ssh-remote+[YOUR_HOST] [PROJECT_PATH]
```

Then in VS Code:

1. Install the VS Code `Dev Containers` extension.
2. Press `F1` and run `Dev Containers: Reopen in Container`.
3. Wait for the initial pull and build to complete.
4. Open a new terminal inside the container.

The devcontainer will:

- build from `.devcontainer/Dockerfile`
- reuse `frankleeeee/sglang-omni:dev` as the base image
- mount the repository at `/sgl-workspace/sglang-omni`
- create a `devuser` user and align container UID/GID with the host when possible

The first time you open the repository in the devcontainer may take longer because Docker needs to pull the base image and build the container image. Once startup is successful, the VS Code status bar should show that you are connected to a dev container.

### Install Python Dependencies Inside The Container

After VS Code finishes attaching to the container, install the project dependencies in the container terminal:

#### install for runtime
```bash
uv pip install -v .
```

#### install for development
```bash
uv pip install -v -e ".[dev]"
```

#### Optional UID/GID Override

If the mounted workspace is not writable inside the container, set the host UID (`id -u`) and GID (`id -u`) in `.devcontainer/devcontainer.json` before rebuilding:

```jsonc
"args": {
  "HOST_UID": "1000",
  "HOST_GID": "1000"
}
```

Then run `Dev Containers: Rebuild Container`.

#### Optional docker args override

You can change Docker runArgs in devcontainer.json

```jsonc
  "runArgs": [
    "--gpus",
    "all",
    "--shm-size",
    "32g"
  ],
```

and devcontainer will run it as

```shell
docker run -it \
    --shm-size 32g \
    --gpus all \
...
```

### Option 2 Start up containers manually
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
