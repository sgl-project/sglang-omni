# 🚀 快速开始 (Get Started)

## 📦 安装指南

我们强烈建议在开发或生产环境中使用我们提供的 [Docker 镜像](#use-docker-zh)。否则，请确保你的环境中已经编译并安装了 [`ucx`](https://github.com/openucx/ucx) 等底层 C++ 通信库。

```bash
# 克隆本仓库
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

# 在系统或 Docker 内使用 uv 创建一个极速虚拟环境
uv venv .venv -p 3.11

# 激活虚拟环境 (Windows 下请使用 .venv\Scripts\activate)
source .venv/bin/activate

# 极速安装
uv pip install -v .

# 以开发模式安装 (代码修改后即时生效)
uv pip install -v -e .
```


(use-docker-zh)=
## 🐳 使用 Docker

我们已经将所有必需的底层依赖（如 NCCL, UCX 等）打包进了我们的 Docker 镜像中，所以你可以直接拉取并运行它，省去繁琐的环境配置。

```bash
# 强烈推荐使用官方镜像以获得最稳定的运行环境
# 注意：该镜像在正式发布后，将被迁移至 lmsysorg 组织下
docker pull frankleeeee/sglang-omni:dev

# 启动容器
# 注意：Relay 数据平面的进程间通信极度依赖于共享内存，请务必分配足够的 --shm-size
docker run -it \
    --shm-size 32g \
    --gpus all \
    --ipc host \
    --network host \
    --privileged \
    frankleeeee/sglang-omni:dev \
    /bin/zsh
```
