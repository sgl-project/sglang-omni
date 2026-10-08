# Node layout shared by setup_node.sh and gate.sh. Every value can be overridden from the environment.
# shellcheck shell=bash disable=SC2034

# Model under test: models/$MODEL.env holds its checkpoint, server, client and unit settings.
MODEL=${MODEL:-minicpmo}
[ -f "$(dirname "${BASH_SOURCE[0]}")/models/$MODEL.env" ] || { echo "config.sh: no models/$MODEL.env" >&2; exit 1; }
# shellcheck source=models/minicpmo.env
source "$(dirname "${BASH_SOURCE[0]}")/models/$MODEL.env"

# Host side.
OMNI_ROOT=${OMNI_ROOT:-$HOME/omni}  # holds models/, logs/, src/ (mounted at /models, /logs, /src)
IMAGE=${IMAGE:-docker.io/lmsysorg/sglang:v0.5.21-cu130}
# Cards to use, the container of each card, and the cores its recorder clients are pinned to (same order).
CARDS=${CARDS:-"0 1 2"}
CONTAINERS=${CONTAINERS:-"sglang-omni-gate sglang-omni-gate-1 sglang-omni-gate-2"}
CLIENT_CPUS_LIST=${CLIENT_CPUS_LIST:-"56-83 112-125,140-153 126-139,154-167"}
UNIT_CPUS=${UNIT_CPUS:-176-223}  # CPU unit tests; keep them off the client cores
PIP_PINS=${PIP_PINS:-"sglang==0.5.21 flashinfer_python[cu13]==0.6.18"}
PIP_EXTRAS=${PIP_EXTRAS:-"onnx openai-whisper gdown pytest pytest-asyncio pytest-timeout"}

# Full-Duplex-Bench v1.0: the five zips of the public Drive folder; the digest is the sha256 of the concatenated per-zip sha256 hex digests.
FDB_FOLDER=${FDB_FOLDER:-https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3}
FDB_DIGEST=${FDB_DIGEST:-cc23e4fc5882f4ac96be5cc048dd50374522d8e25e2318558cdb17220a648bc3}
FDB_ZIPS=${FDB_ZIPS:-"candor_pause_handling.zip:67089249 candor_turn_taking.zip:37850555 icc_backchannel.zip:112509001 synthetic_pause_handling.zip:107106287 synthetic_user_interruption.zip:152934381"}

# Recorder / serving client tree (benchmarks/eval/benchmark_duplex_v10.py and benchmarks/duplex/serving.py), as <tag>:<sha>.
BENCH_TREE=${BENCH_TREE:-bench:88ad7b44ddc920314521d0b2fa1124fbc28bcf32}

# Container side (fixed mount points).
MODELS_DIR=/models
LOGS_DIR=/logs
SRC_DIR=/src
MODEL_DIR=$MODELS_DIR/$MODEL_NAME
FDB_ROOT=$LOGS_DIR/fdb/v1_0_full
HARNESS_DIR=$SRC_DIR/harness
HEADS_DIR=$SRC_DIR/heads
RESULTS_ROOT=$LOGS_DIR/gates
HUB_HOME=$LOGS_DIR/hf-home  # HF_HOME of the servers whose model reads HUB_FILES from the Hub cache
