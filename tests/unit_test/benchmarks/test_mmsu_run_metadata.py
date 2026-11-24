# SPDX-License-Identifier: Apache-2.0
"""MMSU run artifacts describe the requests and selected dataset samples."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer
from datasets import Dataset, DatasetDict
from pydantic import BaseModel

import benchmarks.dataset.mmsu as dataset_module
import benchmarks.eval.benchmark_omni_mmsu as benchmark
from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.mmsu import MmsuSample
from benchmarks.tasks.audio_understanding import DEFAULT_PROMPT


class MmsuMessage(BaseModel):
    role: Literal["user"]
    content: str


class MmsuRequest(BaseModel):
    audios: list[str]
    messages: list[MmsuMessage]
    modalities: list[Literal["text", "audio"]]
    stream: bool
    max_tokens: int
    temperature: float
    seed: int | None = None


@dataclass(kw_only=True)
class MmsuServer:
    base_url: str
    requests: list[MmsuRequest]


@pytest_asyncio.fixture
async def mmsu_server() -> AsyncIterator[MmsuServer]:
    requests: list[MmsuRequest] = []

    async def complete(request: web.Request) -> web.Response:
        payload = MmsuRequest.model_validate(await request.json())
        requests.append(payload)
        audio_paths = payload.audios
        if "timeout.mp3" in audio_paths:
            await asyncio.sleep(0.3)
        else:
            pass
        if any(Path(audio_path).name == "fail.mp3" for audio_path in audio_paths):
            return web.json_response({"error": "synthetic failure"}, status=503)
        else:
            return web.json_response(
                {"choices": [{"message": {"content": "A"}}], "usage": {}}
            )

    application = web.Application()
    application.router.add_post("/v1/chat/completions", complete)
    async with TestServer(application) as server:
        yield MmsuServer(base_url=str(server.make_url("")), requests=requests)


@pytest.fixture
def mmsu_arguments(tmp_path: Path, mmsu_server: MmsuServer) -> argparse.Namespace:
    return argparse.Namespace(
        base_url=mmsu_server.base_url,
        model="synthetic-model",
        modalities="text",
        output_dir=str(tmp_path / "results"),
        max_samples=2,
        task_names=None,
        categories=None,
        repo_id=None,
        prompt=None,
        max_tokens=13,
        temperature=0.2,
        seed=None,
        warmup=None,
        max_concurrency=2,
        request_rate=float("inf"),
        timeout_s=1.25,
        save_audio=False,
        disable_tqdm=True,
        fingerprint=False,
        lang="en",
        asr_device="cpu",
        asr_concurrency=3,
    )


@pytest.fixture
def supplied_samples() -> list[MmsuSample]:
    return [
        MmsuSample(
            sample_id=sample_id,
            audio_path=f"{sample_id}.mp3",
            question="Which note?",
            choices=["A", "B"],
            answer_text="A",
            answer_index=0,
            task_name="pitch",
            category="music",
            sub_category="",
            sub_sub_category="",
            linguistics_sub_discipline="",
        )
        for sample_id in ("normal", "timeout")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("repository", [None, "example/mmsu-subset"])
@pytest.mark.parametrize("seed", [None, 0])
async def test_loaded_dataset_metadata_matches_selection_and_requests(
    monkeypatch: pytest.MonkeyPatch,
    mmsu_arguments: argparse.Namespace,
    mmsu_server: MmsuServer,
    repository: str | None,
    seed: int | None,
) -> None:
    def load_dataset(repo_id: str) -> DatasetDict:
        assert repo_id == (repository or "ddwang2000/MMSU")
        return DatasetDict(
            train=Dataset.from_list(
                [
                    {
                        "id": sample_id,
                        "audio": {"bytes": b"synthetic audio", "path": None},
                        "question": "Which note?",
                        "choice_a": "A",
                        "choice_b": "B",
                        "choice_c": "C",
                        "choice_d": "D",
                        "answer_gt": "A",
                        "task_name": "other" if sample_id == "excluded" else "pitch",
                        "category": "speech" if sample_id == "excluded" else "music",
                        "sub-category": "",
                        "sub-sub-category": "",
                        "linguistics_sub_discipline": "",
                    }
                    for sample_id in ("s0", "s1", "fail", "excluded")
                ]
            )
        )

    monkeypatch.setattr(dataset_module, "load_dataset", load_dataset)
    mmsu_arguments.repo_id = repository
    mmsu_arguments.seed = seed
    mmsu_arguments.task_names = " pitch, ,pitch "
    mmsu_arguments.categories = " music, ,music "
    await benchmark.run(mmsu_arguments)
    artifact = json.loads(
        (Path(mmsu_arguments.output_dir) / "mmsu_results.json").read_text()
    )
    config = artifact["config"]
    sample_ids = ["s0", "s1"] if seed is None else ["s0", "fail"]
    assert config["dataset"] == {
        "source": "huggingface",
        "repo_id": repository or "ddwang2000/MMSU",
        "split": "train",
        "task_names": ["pitch"],
        "categories": ["music"],
        "sampling": (
            "dataset_order_then_limit" if seed is None else "seeded_shuffle_then_limit"
        ),
        "sample_count": 2,
        "sample_ids": sample_ids,
    }
    assert config["timeout_s"] == 1.25
    assert config["warmup"] == 2
    assert config["prompt"] == DEFAULT_PROMPT
    assert config["stream"] is False
    assert config["compute_wer"] is False
    assert config["save_audio"] is False
    assert [sample["sample_id"] for sample in artifact["per_sample"]] == sample_ids
    assert artifact["summary"]["failed_samples"] == (0 if seed is None else 1)
    assert Counter(
        Path(request.audios[0]).stem for request in mmsu_server.requests
    ) == Counter(["s0", "s0", *sample_ids])
    for request in mmsu_server.requests:
        assert request.max_tokens == config["max_tokens"]
        assert request.temperature == config["temperature"]
        assert request.seed == config["seed"]
        assert request.modalities == config["modalities"]
        assert request.stream == config["stream"]
        assert request.messages[0].content.startswith(config["prompt"])


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_provided_sample_metadata_preserves_actual_cohort_and_timeout(
    mmsu_arguments: argparse.Namespace,
    mmsu_server: MmsuServer,
    supplied_samples: list[MmsuSample],
    empty: bool,
) -> None:
    selected_samples = [] if empty else supplied_samples
    mmsu_arguments.repo_id = "ignored/repository"
    mmsu_arguments.task_names = "ignored"
    mmsu_arguments.categories = "ignored"
    mmsu_arguments.seed = 7
    mmsu_arguments.max_samples = 1
    mmsu_arguments.prompt = "Use only the answer letter."
    mmsu_arguments.modalities = "text+audio"
    mmsu_arguments.save_audio = True
    mmsu_arguments.timeout_s = 0.1
    mmsu_arguments.warmup = 0
    await benchmark.run(mmsu_arguments, samples=selected_samples, compute_wer=False)
    artifact = json.loads(
        (Path(mmsu_arguments.output_dir) / "mmsu_results.json").read_text()
    )
    config = artifact["config"]
    assert config["dataset"] == {
        "source": "provided_samples",
        "repo_id": None,
        "split": None,
        "task_names": None,
        "categories": None,
        "sampling": "provided_order",
        "sample_count": len(selected_samples),
        "sample_ids": [sample.sample_id for sample in selected_samples],
    }
    assert config["max_samples"] == 1
    assert config["timeout_s"] == 0.1
    assert config["warmup"] == 0
    assert config["prompt"] == "Use only the answer letter."
    for request in mmsu_server.requests:
        assert request.messages[0].content.startswith(config["prompt"])
        assert request.modalities == config["modalities"]
        assert request.seed == config["seed"]
    assert config["save_audio"] is True
    assert config["compute_wer"] is False
    assert artifact["summary"]["failed_samples"] == (0 if empty else 1)
    assert artifact["summary"]["successful_samples"] == (0 if empty else 1)


@pytest.mark.asyncio
async def test_audio_run_records_wer_inputs(
    monkeypatch: pytest.MonkeyPatch,
    mmsu_arguments: argparse.Namespace,
    supplied_samples: list[MmsuSample],
) -> None:
    def compute_consistency(
        request_results: list[RequestResult],
        lang: str,
        asr_device: str,
        *,
        asr_concurrency: int,
    ) -> dict[str, dict[str, int]]:
        assert len(request_results) == 1
        assert (lang, asr_device, asr_concurrency) == ("zh", "cpu", 3)
        return {"summary": {"samples": 1}}

    monkeypatch.setattr(
        benchmark, "compute_text_audio_consistency", compute_consistency
    )
    mmsu_arguments.modalities = "text+audio"
    mmsu_arguments.lang = "zh"
    for loader_field in ("repo_id", "task_names", "categories"):
        vars(mmsu_arguments).pop(loader_field)
    mmsu_arguments.warmup = 0
    await benchmark.run(mmsu_arguments, samples=supplied_samples[:1])
    artifact = json.loads(
        (Path(mmsu_arguments.output_dir) / "mmsu_results.json").read_text()
    )
    config = artifact["config"]
    assert config["compute_wer"] is True
    assert config["asr_concurrency"] == 3
    assert config["asr_device"] == "cpu"
    assert config["lang"] == "zh"
    assert artifact["wer"] == {"summary": {"samples": 1}}
