# SPDX-License-Identifier: Apache-2.0
"""Prepares a CI data root: pinned model checkpoints and the frozen corpus.

    python provision.py <data_root> [--corpus] [--model REPO ...]

Checkpoints and the corpus's source files come from the Hugging Face Hub. Every
file is checked against its SHA-256 in model_pins.json, corpus/sources.json or
corpus/v1.sha256, so a runner's cache is reused only when it matches what the
golden files were made from. A checkpoint lands in <data_root>/models/<org>_<name>;
a Whisper checkpoint also gets the tokenizer files of its openai/whisper-* repo
in that directory, where Voxt puts them, and Fun-CosyVoice3 gets its S3 speech
tokenizer's checkpoint in speech_tokenizer/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from huggingface_hub import snapshot_download

CI_DIRECTORY = Path(__file__).resolve().parent
DOWNLOAD_CHUNK_BYTES = 1 << 20
BYTES_PER_MEGABYTE = 1_000_000


class FileSource(TypedDict):
    revision: str
    files: dict[str, str]


class TokenizerSource(FileSource):
    repo: str
    # Where under the model directory its files go; the model directory itself
    # when absent.
    subdirectory: NotRequired[str]


class ModelPin(FileSource):
    tokenizer: NotRequired[TokenizerSource]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_hashes(path: Path) -> dict[str, str]:
    hashes = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            hashes[name.strip()] = digest
        else:
            pass
    return hashes


def download(url: str, destination: Path, expected_sha256: str) -> None:
    """Downloads url to destination unless a file with the expected hash is there."""
    if destination.exists() and sha256(destination) == expected_sha256:
        return
    else:
        pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    print(f"downloading {url}", flush=True)
    with (
        urllib.request.urlopen(url, timeout=600) as response,
        partial.open("wb") as out,
    ):
        shutil.copyfileobj(response, out, DOWNLOAD_CHUNK_BYTES)
    actual = sha256(partial)
    if actual != expected_sha256:
        partial.unlink()
        raise SystemExit(
            f"{url}: SHA-256 {actual} does not match the pinned {expected_sha256}"
        )
    else:
        pass
    partial.rename(destination)


def provision_files(
    repo: str,
    source: FileSource,
    directory: Path,
    *,
    repo_type: Literal["model", "dataset"],
) -> None:
    """Fetches the pinned files of repo with huggingface_hub unless their SHA-256 already matches."""
    stale_names = [
        name
        for name, digest in source["files"].items()
        if not (directory / name).exists() or sha256(directory / name) != digest
    ]
    if stale_names:
        # Note (XXXXRT666): huggingface_hub keeps a file whose recorded revision matches without rehashing it, so a mismatching file must be gone first.
        for name in stale_names:
            (directory / name).unlink(missing_ok=True)
        print(
            f"downloading {len(stale_names)} files of {repo} at {source['revision']}",
            flush=True,
        )
        start_seconds = time.monotonic()
        snapshot_download(
            repo,
            repo_type=repo_type,
            revision=source["revision"],
            allow_patterns=stale_names,
            local_dir=directory,
        )
        elapsed_seconds = time.monotonic() - start_seconds
        downloaded_megabytes = (
            sum((directory / name).stat().st_size for name in stale_names)
            / BYTES_PER_MEGABYTE
        )
        print(
            f"downloaded {downloaded_megabytes:.0f} MB in {elapsed_seconds:.0f} s"
            f" ({downloaded_megabytes / elapsed_seconds:.1f} MB/s)",
            flush=True,
        )
        for name in stale_names:
            actual = sha256(directory / name)
            if actual != source["files"][name]:
                raise SystemExit(
                    f"{repo}/{name}: SHA-256 {actual} does not match the pinned {source['files'][name]}"
                )
            else:
                pass
    else:
        pass


def provision_model(data_root: Path, repo: str, pin: ModelPin) -> Path:
    model_directory = data_root / "models" / repo.replace("/", "_")
    provision_files(repo, pin, model_directory, repo_type="model")
    if "tokenizer" in pin:
        tokenizer = pin["tokenizer"]
        provision_files(
            tokenizer["repo"],
            tokenizer,
            model_directory / tokenizer.get("subdirectory", ""),
            repo_type="model",
        )
    else:
        pass
    return model_directory


def extract(archive_path: Path, extracted: Path, top_directory: str) -> None:
    """Extracts the archive's top_directory into extracted unless it is there.

    The archive is unpacked beside it first, so a run killed midway leaves no half-extracted directory that a later run would take as complete.
    """
    if (extracted / top_directory).exists():
        pass
    else:
        staging = extracted / f".{top_directory}.partial"
        shutil.rmtree(staging, ignore_errors=True)
        with tarfile.open(archive_path) as archive:
            archive.extractall(staging, filter="data")
        (staging / top_directory).rename(extracted / top_directory)
        shutil.rmtree(staging)


def corpus_is_complete(corpus_directory: Path, hashes: dict[str, str]) -> bool:
    return all(
        (corpus_directory / name).exists() and sha256(corpus_directory / name) == digest
        for name, digest in hashes.items()
    )


def provision_corpus(data_root: Path) -> Path:
    corpus_root = data_root / "corpus"
    corpus_directory = corpus_root / "v1"
    clip_hashes = read_hashes(CI_DIRECTORY / "corpus" / "v1.sha256")
    if corpus_is_complete(corpus_directory, clip_hashes):
        return corpus_directory
    else:
        pass
    # Note (XXXXRT666): the sources stay under raw/ until the clips are built, so a run that times out leaves its downloads for the next one.
    sources: dict[str, FileSource] = json.loads(
        (CI_DIRECTORY / "corpus" / "sources.json").read_text()
    )
    raw = corpus_root / "raw"
    for repo, source in sources.items():
        provision_files(repo, source, raw / repo.replace("/", "_"), repo_type="dataset")
    librispeech = raw / "k2-fsa_LibriSpeech"
    fleurs = raw / "google_fleurs" / "data" / "cmn_hans_cn"
    extracted = corpus_root / "extracted"
    extract(librispeech / "test-clean.tar.gz", extracted, "LibriSpeech")
    extract(fleurs / "audio" / "test.tar.gz", extracted, "test")
    subprocess.run(
        [
            sys.executable,
            str(CI_DIRECTORY / "corpus" / "build_corpus.py"),
            str(data_root),
            str(fleurs / "test.tsv"),
            str(raw / "CAiRE_ASCEND" / "main" / "test-00000-of-00001.parquet"),
        ],
        check=True,
    )
    if not corpus_is_complete(corpus_directory, clip_hashes):
        raise SystemExit("the rebuilt corpus does not match corpus/v1.sha256")
    else:
        pass
    # Note (Jiaxin Deng): later runs read only the clips; the archives and their
    # extracted audio (about 2 GB) are only needed to rebuild them.
    shutil.rmtree(corpus_root / "raw", ignore_errors=True)
    shutil.rmtree(extracted, ignore_errors=True)
    return corpus_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--corpus", action="store_true", help="also the corpus")
    parser.add_argument(
        "--model", action="append", default=[], help="pinned repo to provision"
    )
    arguments = parser.parse_args()
    pins: dict[str, ModelPin] = json.loads(
        (CI_DIRECTORY / "model_pins.json").read_text()
    )
    for repo in arguments.model:
        print(
            f"model {repo}: {provision_model(arguments.data_root, repo, pins[repo])}",
            flush=True,
        )
    if arguments.corpus:
        print(f"corpus: {provision_corpus(arguments.data_root)}", flush=True)
    else:
        pass


if __name__ == "__main__":
    main()
