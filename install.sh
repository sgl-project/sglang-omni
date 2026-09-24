#!/usr/bin/env bash
# Install sglang-omni and its Apple Silicon runtime prerequisites.
#
# The script is intentionally self-contained so it can be used as:
#
#   curl -fsSL https://raw.githubusercontent.com/sgl-project/sglang-omni/main/install.sh | bash
#
# It installs the shared Homebrew prerequisites and the currently validated
# Apple audio model prerequisites by default, so one command is enough for
# Qwen3-ASR, Qwen3-TTS, Whisper/Fun-ASR, and Fun-CosyVoice3.
set -Eeuo pipefail
IFS=$'\n\t'

readonly DEFAULT_OMNI_REPO="https://github.com/sgl-project/sglang-omni.git"
readonly DEFAULT_OMNI_REF="main"
readonly DEFAULT_SGLANG_REPO="https://github.com/sgl-project/sglang.git"
readonly DEFAULT_SGLANG_REF="v0.5.20"

SCRIPT_PATH="${BASH_SOURCE[0]:-}"
if [[ -n "$SCRIPT_PATH" && -f "$SCRIPT_PATH" ]]; then
  readonly SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)"
else
  readonly SCRIPT_DIR=""
fi

readonly OMNI_REPO="${SGLANG_OMNI_REPO:-$DEFAULT_OMNI_REPO}"
readonly OMNI_REF="${SGLANG_OMNI_REF:-$DEFAULT_OMNI_REF}"
readonly SGLANG_REPO="${SGLANG_REPO:-$DEFAULT_SGLANG_REPO}"
readonly SGLANG_REF="${SGLANG_VERSION:-$DEFAULT_SGLANG_REF}"
readonly CACHE_ROOT="${SGLANG_OMNI_CACHE:-${HOME}/.cache/sglang-omni}"
readonly SGLANG_DIR="${SGLANG_SOURCE_DIR:-${CACHE_ROOT}/sglang-${SGLANG_REF//\//-}}"
readonly OMNI_DIR="${SGLANG_OMNI_PROJECT_DIR:-${CACHE_ROOT}/sglang-omni-${OMNI_REF//\//-}}"
readonly COSYVOICE_DIR="${SGLANG_OMNI_COSYVOICE_DIR:-${CACHE_ROOT}/CosyVoice}"
readonly BOOTSTRAP_HOMEBREW="${SGLANG_OMNI_BOOTSTRAP_HOMEBREW:-1}"

NONINTERACTIVE="${NONINTERACTIVE:-0}"
REQUESTED_EXTRAS="$(IFS=,; printf '%s' "${APPLE_OMNI_EXTRAS[*]}")${SGLANG_OMNI_EXTRAS:+,$SGLANG_OMNI_EXTRAS}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"

log() {
  printf '[sglang-omni-install] %s\n' "$*"
}

die() {
  printf '[sglang-omni-install][error] %s\n' "$*" >&2
  exit 1
}

on_error() {
  local exit_code=$?
  printf '[sglang-omni-install][error] command failed at line %s: %s\n' "$1" "$2" >&2
  exit "$exit_code"
}

trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

SGLANG_STAGE_DIR=""
CHECKOUT_TMP_DIR=""
cleanup() {
  if [[ -n "$SGLANG_STAGE_DIR" && -d "$SGLANG_STAGE_DIR" ]]; then
    rm -rf -- "$SGLANG_STAGE_DIR"
  fi
  if [[ -n "$CHECKOUT_TMP_DIR" && -d "$CHECKOUT_TMP_DIR" ]]; then
    rm -rf -- "$CHECKOUT_TMP_DIR"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

usage() {
  cat <<'EOF'
Usage: ./install.sh [OPTIONS]

Install sglang-omni and the currently validated Apple audio model
prerequisites into an isolated Python 3.12 virtual environment. The one
command setup covers Qwen3-ASR, Qwen3-TTS, Whisper/Fun-ASR, and
Fun-CosyVoice3.

Options:
  --non-interactive   Disable Homebrew auto-update and pass NONINTERACTIVE to
                      Homebrew's bootstrapper (CI use; sudo may still be needed).
  -h, --help          Show this help.

Environment:
  SGLANG_OMNI_VENV         Virtual environment path.
  SGLANG_OMNI_CACHE        Cache root for source checkouts.
  SGLANG_SOURCE_DIR        SGLang source checkout path.
  SGLANG_VERSION            SGLang git tag/branch (default: v0.5.20).
  SGLANG_REPO               SGLang repository URL.
  SGLANG_OMNI_REPO          sglang-omni repository URL for hosted use.
  SGLANG_OMNI_REF           sglang-omni branch/tag (default: main).
  SGLANG_OMNI_PROJECT_DIR   sglang-omni checkout path for hosted use.
  SGLANG_OMNI_EXTRAS        Optional extras, comma-separated.
  NONINTERACTIVE=1          Same as --non-interactive.
  UV_HTTP_TIMEOUT            Per-request timeout in seconds (default: 300).
  UV_HTTP_RETRIES            Network retry count (default: 5).

The local checkout is installed when this script lives in an sglang-omni
repository. If downloaded or piped from stdin, the configured repository/ref
is cloned into SGLANG_OMNI_PROJECT_DIR (or the cache default above).
EOF
}

while (($#)); do
  case "$1" in
    --non-interactive)
      NONINTERACTIVE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
  shift
done

[[ "$(uname -s)" == "Darwin" ]] || die "unsupported operating system; this installer requires macOS"
[[ "$(uname -m)" == "arm64" ]] || die "unsupported architecture; Apple Silicon arm64 is required"

if [[ -n "${SGLANG_OMNI_PROJECT_DIR:-}" ]]; then
  # An explicit destination selects hosted mode even when a local script copy
  # happens to sit next to a checkout.
  PROJECT_DIR=""
elif [[ -n "$SCRIPT_DIR" && -f "$SCRIPT_DIR/pyproject.toml" ]] \
  && grep -q '^name = "sglang-omni"' "$SCRIPT_DIR/pyproject.toml"; then
  PROJECT_DIR="$SCRIPT_DIR"
elif [[ -f "$PWD/pyproject.toml" ]] && grep -q '^name = "sglang-omni"' "$PWD/pyproject.toml"; then
  # A downloaded installer can still be run from a checked-out repository.
  PROJECT_DIR="$PWD"
else
  PROJECT_DIR=""
fi

if [[ -n "${SGLANG_OMNI_VENV:-}" ]]; then
  readonly VENV_DIR="$SGLANG_OMNI_VENV"
elif [[ -n "$PROJECT_DIR" ]]; then
  readonly VENV_DIR="$PROJECT_DIR/.venv-apple"
else
  readonly VENV_DIR="$CACHE_ROOT/.venv-apple"
fi

find_brew() {
  local candidate
  # Prefer native Apple Silicon Homebrew even if an Intel/Rosetta brew appears
  # earlier on PATH.
  if [[ -x /opt/homebrew/bin/brew ]]; then
    printf '%s\n' /opt/homebrew/bin/brew
    return 0
  fi
  if command -v brew >/dev/null 2>&1; then
    command -v brew
    return 0
  fi
  for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

BREW_BIN="$(find_brew || true)"
if [[ -z "$BREW_BIN" ]]; then
  [[ "$BOOTSTRAP_HOMEBREW" != "0" ]] \
    || die "Homebrew is not installed; unset SGLANG_OMNI_BOOTSTRAP_HOMEBREW=0 to allow the official bootstrapper"
  command -v curl >/dev/null 2>&1 || die "curl is required to bootstrap Homebrew"
  log "Homebrew was not found; running the official Homebrew bootstrapper"
  HOMEBREW_INSTALLER="$(curl -fsSL "$HOMEBREW_INSTALL_URL")" \
    || die "could not download the official Homebrew bootstrapper"
  if [[ "$NONINTERACTIVE" == "1" ]]; then
    NONINTERACTIVE=1 /bin/bash -c "$HOMEBREW_INSTALLER"
  else
    /bin/bash -c "$HOMEBREW_INSTALLER"
  fi
  BREW_BIN="$(find_brew || true)"
  [[ -n "$BREW_BIN" ]] || die "Homebrew bootstrap completed but /opt/homebrew/bin/brew was not found"
fi

BREW_PREFIX="$($BREW_BIN --prefix)" || die "unable to determine the Homebrew prefix"
[[ "$BREW_PREFIX" == "/opt/homebrew" ]] \
  || die "native Apple Silicon Homebrew is required; found prefix: $BREW_PREFIX"
export PATH="$BREW_PREFIX/bin:$BREW_PREFIX/sbin:$PATH"
if [[ "$NONINTERACTIVE" == "1" ]]; then
  export HOMEBREW_NO_AUTO_UPDATE="${HOMEBREW_NO_AUTO_UPDATE:-1}"
fi

ensure_formula() {
  local formula="$1"
  if "$BREW_BIN" list --formula "$formula" >/dev/null 2>&1; then
    log "Homebrew formula already installed: $formula"
    return 0
  fi
  log "Installing Homebrew formula: $formula"
  if ! "$BREW_BIN" install "$formula"; then
    die "Homebrew could not install $formula. Check 'brew doctor' and Homebrew directory permissions; no administrator command was run by this script"
  fi
}

ensure_formula ffmpeg@7
ensure_formula uv

if ! command -v git >/dev/null 2>&1 || ! git --version >/dev/null 2>&1; then
  ensure_formula git
fi
command -v git >/dev/null 2>&1 || die "git is required; install it with Homebrew or Xcode Command Line Tools"
command -v uv >/dev/null 2>&1 || die "uv is required; the Homebrew uv formula was not found on PATH"

if [[ -d "$VENV_DIR" ]]; then
  log "Reusing Python virtual environment: $VENV_DIR"
  PYTHON_BIN="$VENV_DIR/bin/python"
  [[ -x "$PYTHON_BIN" ]] || die "existing path is not a usable virtual environment: $VENV_DIR"
else
  log "Creating Python 3.12 virtual environment: $VENV_DIR"
  uv venv --python 3.12 "$VENV_DIR"
  PYTHON_BIN="$VENV_DIR/bin/python"
fi
[[ -x "$PYTHON_BIN" ]] || die "uv did not create $PYTHON_BIN"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || die "the virtual environment must use Python 3.12"
UV_PIP_INSTALL=(uv pip install --python "$PYTHON_BIN")

fetch_checkout_ref() {
  local destination="$1"
  local ref="$2"
  local fetch_spec="$ref"
  local tag_ref remote_tag

  case "$ref" in
    refs/tags/*)
      fetch_spec="$ref:$ref"
      ;;
    refs/*)
      ;;
    *)
      # Full object IDs must keep their meaning even if a tag has the same name.
      if [[ ! "$ref" =~ ^[[:xdigit:]]{40}$ && ! "$ref" =~ ^[[:xdigit:]]{64}$ ]]; then
        tag_ref="refs/tags/$ref"
        remote_tag="$(git -C "$destination" ls-remote --refs origin "$tag_ref" \
          | awk -v ref="$tag_ref" '$2 == ref { print $2 }')"
        if [[ -n "$remote_tag" ]]; then
          fetch_spec="$tag_ref:$tag_ref"
        fi
      fi
      ;;
  esac
  # FETCH_HEAD alone does not preserve the release tag used by setuptools-scm.
  git -C "$destination" fetch --depth 1 origin "$fetch_spec"
}

clone_or_reuse() {
  local destination="$1"
  local ref="$2"
  local repository="$3"
  local label="$4"
  local actual_repository current_commit target_commit

  if [[ -e "$destination" && ! -d "$destination/.git" ]]; then
    die "$label destination exists but is not a git checkout: $destination (choose another path)"
  fi
  if [[ -d "$destination/.git" ]]; then
    actual_repository="$(git -C "$destination" remote get-url origin 2>/dev/null || true)"
    [[ "$actual_repository" == "$repository" ]] \
      || die "$label checkout origin is $actual_repository, expected $repository: $destination"
    [[ -z "$(git -C "$destination" status --porcelain)" ]] \
      || die "$label checkout has local changes; clean it or choose another path: $destination"
    log "Refreshing $label $ref: $destination"
    fetch_checkout_ref "$destination" "$ref"
    current_commit="$(git -C "$destination" rev-parse HEAD)"
    target_commit="$(git -C "$destination" rev-parse 'FETCH_HEAD^{commit}')"
    if [[ "$current_commit" != "$target_commit" ]]; then
      log "Updating $label checkout to $target_commit"
      git -C "$destination" checkout --detach --quiet FETCH_HEAD
    fi
    return 0
  fi
  log "Cloning $label $ref: $destination"
  mkdir -p "$(dirname -- "$destination")"
  CHECKOUT_TMP_DIR="$(mktemp -d "${destination}.tmp.XXXXXX")"
  git -C "$CHECKOUT_TMP_DIR" init --quiet
  git -C "$CHECKOUT_TMP_DIR" remote add origin "$repository"
  fetch_checkout_ref "$CHECKOUT_TMP_DIR" "$ref"
  git -C "$CHECKOUT_TMP_DIR" checkout --detach --quiet FETCH_HEAD
  [[ ! -e "$destination" ]] || die "$label destination appeared during installation: $destination"
  mv "$CHECKOUT_TMP_DIR" "$destination"
  CHECKOUT_TMP_DIR=""
}

clone_or_reuse "$SGLANG_DIR" "$SGLANG_REF" "$SGLANG_REPO" "SGLang"

# SGLang keeps the CUDA-oriented project metadata in pyproject.toml and the
# platform-neutral/MLX metadata in pyproject_other.toml. Stage the whole
# checkout and replace only the staged metadata, leaving the source untouched.
SGLANG_STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sglang-omni-sglang.XXXXXX")"
cp -R "$SGLANG_DIR/." "$SGLANG_STAGE_DIR/"
if [[ -f "$SGLANG_DIR/python/pyproject_other.toml" ]]; then
  cp "$SGLANG_DIR/python/pyproject_other.toml" "$SGLANG_STAGE_DIR/python/pyproject.toml"
else
  die "SGLang checkout does not contain python/pyproject_other.toml"
fi

log "Installing SGLang $SGLANG_REF with the all_mps extra"
# The optional SGLang Rust extensions are not used by the MLX/Torch-MPS path.
SGLANG_BUILD_RUST_EXTS=none \
  "${UV_PIP_INSTALL[@]}" --prerelease=allow "$SGLANG_STAGE_DIR/python[all_mps]"

if [[ -z "$PROJECT_DIR" ]]; then
  PROJECT_DIR="$OMNI_DIR"
  clone_or_reuse "$PROJECT_DIR" "$OMNI_REF" "$OMNI_REPO" "sglang-omni"
fi
[[ -f "$PROJECT_DIR/pyproject.toml" ]] || die "sglang-omni pyproject.toml not found in $PROJECT_DIR"
grep -q '^name = "sglang-omni"' "$PROJECT_DIR/pyproject.toml" \
  || die "project is not sglang-omni: $PROJECT_DIR"

PROJECT_SPEC="$PROJECT_DIR"
if [[ -n "$REQUESTED_EXTRAS" ]]; then
  PROJECT_SPEC="${PROJECT_DIR}[${REQUESTED_EXTRAS}]"
fi
log "Installing sglang-omni${REQUESTED_EXTRAS:+ with extras: $REQUESTED_EXTRAS}"
"${UV_PIP_INSTALL[@]}" --prerelease=allow -e "$PROJECT_SPEC"

verify_dependency_graph() {
  local check_output check_status=0 bad_lines allowed_lines allowed_count
  local summary_count found_count
  trap - ERR
  check_output="$(uv pip check --python "$PYTHON_BIN" 2>&1)" || check_status=$?
  trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
  if (( check_status == 0 )); then
    printf '%s\n' "$check_output"
    return 0
  fi
  # qwen-tts is deliberately installed with --no-deps because its metadata
  # pins Transformers/Accelerate versions that conflict with Omni. On reruns,
  # accept only those two known qwen-tts conflicts and fail on everything else.
  allowed_lines="$(printf '%s\n' "$check_output" \
    | grep -E '^The package `qwen-tts` requires `(transformers|accelerate)==[^`]+`, but `[^`]+` is installed$' || true)"
  allowed_count="$(printf '%s\n' "$allowed_lines" | sed '/^$/d' | wc -l | tr -d ' ')"
  summary_count="$(printf '%s\n' "$check_output" \
    | grep -Ec '^Using Python .+ environment at: .+$|^Checked [0-9]+ packages in .+$|^Found [0-9]+ incompatibilities?$' || true)"
  found_count="$(printf '%s\n' "$check_output" \
    | sed -nE 's/^Found ([0-9]+) incompatibilities?$/\1/p')"
  bad_lines="$(printf '%s\n' "$check_output" \
    | grep -Ev '^$|^Using Python .+ environment at: .+$|^Checked [0-9]+ packages in .+$|^Found [0-9]+ incompatibilities?$|^The package `qwen-tts` requires `(transformers|accelerate)==[^`]+`, but `[^`]+` is installed$' || true)"
  if [[ -n "$bad_lines" || "$allowed_count" != "2" \
    || "$summary_count" != "3" || "$found_count" != "$allowed_count" ]]; then
    [[ -z "$bad_lines" ]] || printf '%s\n' "$bad_lines" >&2
    die "the installed Omni dependency graph is inconsistent"
  fi
  log "Accepting qwen-tts metadata conflicts from its intentional --no-deps install"
  printf '%s\n' "$check_output"
}

verify_dependency_graph
# qwen-tts declares an incompatible Transformers pin; install only the
# package code and its small direct helper, keeping the Omni stack intact.
log "Installing Qwen3-TTS model package without replacing the Omni stack"
"${UV_PIP_INSTALL[@]}" --prerelease=allow --no-deps "${APPLE_NO_DEPS_PACKAGES[@]}"

FFMPEG_PREFIX="$($BREW_BIN --prefix ffmpeg@7)"
FFMPEG_BIN="$FFMPEG_PREFIX/bin"
FFMPEG_LIB="$FFMPEG_PREFIX/lib"
[[ -d "$FFMPEG_LIB" ]] || die "ffmpeg@7 library directory not found: $FFMPEG_LIB"

log "Verifying the installed Python package and CLI"
"$PYTHON_BIN" -c 'import sglang_omni; print("sglang_omni", sglang_omni.__version__)'
[[ -x "$VENV_DIR/bin/sgl-omni" ]] || die "sgl-omni console script was not installed"
"$VENV_DIR/bin/sgl-omni" --help >/dev/null

# Importing AudioDecoder alone does not load FFmpeg. Instantiate it with a
# tiny in-memory WAV so the dynamic loader is exercised during installation.
DYLD_LIBRARY_PATH="$FFMPEG_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
PATH="$FFMPEG_BIN:$PATH" SGLANG_USE_MLX=1 "$PYTHON_BIN" - <<'PY'
import io
import struct
import wave

import mlx.core as mx
from torchcodec.decoders import AudioDecoder

assert mx.metal.is_available(), "MLX Metal is unavailable"
with io.BytesIO() as stream:
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(struct.pack("<h", 0) * 160)
    decoder = AudioDecoder(stream.getvalue())
    assert decoder.metadata.duration_seconds > 0
print("MLX Metal and TorchCodec audio decoding are available")
PY

cat <<EOF

Installation complete.
Virtual environment: $VENV_DIR
Activate it with:
  source "$VENV_DIR/bin/activate"

For TorchCodec/FFmpeg audio decoding, export:
  export DYLD_LIBRARY_PATH="$FFMPEG_LIB\${DYLD_LIBRARY_PATH:+:\$DYLD_LIBRARY_PATH}"

The Apple Silicon Qwen3-ASR examples are documented at:
  $PROJECT_DIR/docs/cookbook/qwen3_asr.md
EOF
