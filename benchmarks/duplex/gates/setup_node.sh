#!/bin/bash
# setup_node.sh <deps-tag>: prepare a GPU node for the gates. Runs on the node host, idempotent, about 15 minutes on a fresh node.
#   1. pull $IMAGE;
#   2. download $MODEL_REPO@$MODEL_REVISION of models/$MODEL.env into $OMNI_ROOT/models and check every file against its Hub digest
#      (HF_TOKEN, when set, authorizes gated checkpoints such as PersonaPlex; it is used only here);
#   3. create one container per card in $CARDS (--device nvidia.com/gpu=<UUID>; a single container with CUDA_VISIBLE_DEVICES breaks stage platform resolution);
#      the first is provisioned (apt sox/unzip, $PIP_PINS, the dependencies of the pushed tree <deps-tag>, $PIP_EXTRAS), committed, and cloned for the other cards;
#   4. fetch the HUB_FILES of the model (VoiceChat: the tokenizer and config of its backbone repo) into the HF cache $OMNI_ROOT/logs/hf-home;
#   5. fetch Full-Duplex-Bench v1.0 (only the five v1.0 zips) into $OMNI_ROOT/logs/fdb and check $FDB_DIGEST.
# Run it once per model (MODEL=personaplex setup_node.sh base); the steps already done are skipped.
# Push the harness and the <deps-tag> tree first (gate.sh push). Run detached, e.g.
#   setsid nohup bash ~/omni/src/harness/setup_node.sh base > ~/omni/setup.log 2>&1 < /dev/null &
# and wait for the line SETUP_DONE (or SETUP_FAILED). MPS is started per container by gate.sh, not here.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=config.sh source-path=SCRIPTDIR
source "$HERE/config.sh"
DEPS_TAG=${1:?usage: setup_node.sh <deps-tag>}
read -r -a cards <<< "$CARDS"
read -r -a containers <<< "$CONTAINERS"
fail() { echo "SETUP_FAILED: $*"; exit 1; }

mkdir -p "$OMNI_ROOT/models" "$OMNI_ROOT/logs/fdb" "$OMNI_ROOT/src/heads"
[ -f "$OMNI_ROOT/src/heads/$DEPS_TAG.tgz" ] || fail "no $OMNI_ROOT/src/heads/$DEPS_TAG.tgz (run gate.sh push first)"
[ ${#cards[@]} = ${#containers[@]} ] || fail "CARDS and CONTAINERS differ in length"

echo "$(date -u +%T) pulling $IMAGE"
podman pull -q "$IMAGE" > "$OMNI_ROOT/logs/pull.log" 2>&1 &
pull_pid=$!

echo "$(date -u +%T) model $MODEL_REPO@$MODEL_REVISION"
python3 - "$MODEL_REPO" "$MODEL_REVISION" "$OMNI_ROOT/models/$MODEL_NAME" <<'PY' || fail "model download"
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

repo, revision, out = sys.argv[1:]
token = os.environ.get("HF_TOKEN")
auth = {"Authorization": f"Bearer {token}"} if token else {}
tree_url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1"
entries = [e for e in json.load(urllib.request.urlopen(urllib.request.Request(tree_url, headers=auth))) if e["type"] == "file"]


def digest_ok(entry):
    path = os.path.join(out, entry["path"])
    if not os.path.isfile(path) or os.path.getsize(path) != entry["size"]:
        return False
    if "lfs" in entry:
        h = hashlib.sha256()
        expected = entry["lfs"]["oid"]
    else:
        h = hashlib.sha1(f"blob {entry['size']}\0".encode())
        expected = entry["oid"]
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest() == expected


def fetch(entry):
    path = os.path.join(out, entry["path"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for _ in range(3):
        if digest_ok(entry):
            return None
        if os.path.isfile(path) and os.path.getsize(path) >= entry["size"]:
            os.remove(path)  # complete but wrong: download again instead of resuming
        else:
            pass  # missing or partial: curl -C - resumes it
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{entry['path']}"
        header = [arg for key, value in auth.items() for arg in ("-H", f"{key}: {value}")]
        subprocess.run(["curl", "-sfL", "--retry", "5", *header, "-C", "-", url, "-o", path], check=False)
    return None if digest_ok(entry) else entry["path"]


with concurrent.futures.ThreadPoolExecutor(8) as pool:
    bad = [p for p in pool.map(fetch, entries) if p]
total = sum(e["size"] for e in entries)
print(f"model files={len(entries)} bytes={total} digest_mismatch={len(bad)} {bad[:5]}")
sys.exit(1 if bad else 0)
PY

wait $pull_pid || fail "image pull: $(tail -2 "$OMNI_ROOT/logs/pull.log")"
echo "$(date -u +%T) image $IMAGE pulled"

create() { # <name> <card> <image>
  if podman container exists "$1"; then
    echo "container $1 exists"
    return 0
  fi
  local uuid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$2") || return 1
  podman run -d --name "$1" --device "nvidia.com/gpu=$uuid" --ipc=host --network host \
    -v "$OMNI_ROOT/models:$MODELS_DIR:ro" -v "$OMNI_ROOT/logs:$LOGS_DIR" -v "$OMNI_ROOT/src:$SRC_DIR" "$3" sleep infinity > /dev/null || return 1
  podman exec "$1" bash -c 'rm -rf /tmp/mps-* /tmp/sglang_omni; nvidia-smi --query-gpu=uuid --format=csv,noheader' | sed "s/^/$1 card $2: /"
}

first=${containers[0]}
if ! podman container exists "$first"; then
  create "$first" "${cards[0]}" "$IMAGE" || fail "create $first"
  echo "$(date -u +%T) provisioning $first"
  # shellcheck disable=SC2016  # expands inside the container
  podman exec -e PIP_PINS="$PIP_PINS" -e PIP_EXTRAS="$PIP_EXTRAS" -e DEPS_TAG="$DEPS_TAG" "$first" bash -c '
    set -e
    (apt-get update -qq && apt-get install -y -qq sox unzip) > /dev/null 2>&1
    # shellcheck disable=SC2086
    pip install -q $PIP_PINS 2>&1 | tail -3
    rm -rf /tmp/deps && mkdir -p /tmp/deps && tar -xzf /src/heads/$DEPS_TAG.tgz -C /tmp/deps
    pip install -q /tmp/deps 2>&1 | tail -3
    pip uninstall -y -q sglang-omni
    rm -rf /tmp/deps
    # shellcheck disable=SC2086
    pip install -q $PIP_EXTRAS 2>&1 | tail -3
    python -c "import sglang, torch, transformers, flashinfer; print(\"VERSIONS sglang\", sglang.__version__, \"torch\", torch.__version__, \"transformers\", transformers.__version__, \"flashinfer\", flashinfer.__version__)"
    pip list 2>/dev/null > /logs/pip-list.txt' || fail "provision $first"
fi
env_image=localhost/$first-env:latest
podman image exists "$env_image" || podman commit -q "$first" "$env_image" > /dev/null || fail "commit $first"
for i in "${!cards[@]}"; do
  [ "$i" = 0 ] && continue
  create "${containers[$i]}" "${cards[$i]}" "$env_image" || fail "create ${containers[$i]}"
done

if [ -n "$HUB_FILES" ]; then
  echo "$(date -u +%T) Hub files $HUB_FILES"
  # shellcheck disable=SC2016  # expands inside the container
  podman exec -e HUB_FILES="$HUB_FILES" -e HF_HOME="$HUB_HOME" -e HF_TOKEN="${HF_TOKEN:-}" "$first" python -c '
import os
from huggingface_hub import hf_hub_download
for entry in os.environ["HUB_FILES"].split():
    spec, files = entry.split(":")
    repo, revision = spec.split("@")
    for name in files.split(","):
        print(hf_hub_download(repo, name, revision=revision, token=os.environ["HF_TOKEN"] or None))
    # the server resolves the default revision offline (HF_HUB_OFFLINE=1): point it at the pinned one
    refs = os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo.replace("/", "--"), "refs")
    os.makedirs(refs, exist_ok=True)
    open(os.path.join(refs, "main"), "w").write(revision)' || fail "Hub files"
fi

echo "$(date -u +%T) Full-Duplex-Bench v1.0"
# shellcheck disable=SC2016
podman exec -e FDB_FOLDER="$FDB_FOLDER" -e FDB_ZIPS="$FDB_ZIPS" -e FDB_DIGEST="$FDB_DIGEST" "$first" bash -c '
  cd /logs/fdb && mkdir -p drive v1_0_full
  complete() { local ok=0 w; for w in $FDB_ZIPS; do [ "$(stat -c %s "drive/v1.0/${w%%:*}" 2>/dev/null)" = "${w#*:}" ] && ok=$((ok + 1)); done; [ $ok = 5 ]; }
  if ! complete; then
    # the folder also holds v1.5 / v3.0, whose download stalls: stop gdown once the five v1.0 zips are complete
    setsid gdown --folder "$FDB_FOLDER" -O /logs/fdb/drive > /logs/fdb/gdown.log 2>&1 < /dev/null &
    gdown_pid=$!
    for _ in $(seq 1 360); do complete && break; sleep 5; done
    kill -- -"$gdown_pid" 2> /dev/null
    rm -rf drive/v1.5 drive/v3.0
  fi
  complete || { echo "FDB zips incomplete"; exit 1; }
  digest=$(sha256sum drive/v1.0/*.zip | cut -c1-64 | sha256sum | cut -c1-64)
  [ "$digest" = "$FDB_DIGEST" ] || { echo "FDB digest $digest != $FDB_DIGEST"; exit 1; }
  for z in drive/v1.0/*.zip; do unzip -q -o "$z" -d v1_0_full; done
  echo "FDB digest ok; samples per subset: $(for d in v1_0_full/*/; do echo -n "$(basename "$d")=$(ls "$d" | grep -cE "^[0-9]+$") "; done)"' || fail "FDB v1.0"
echo "$(date -u +%T) SETUP_DONE containers: ${containers[*]}"
