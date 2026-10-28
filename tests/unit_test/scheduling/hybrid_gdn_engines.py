# SPDX-License-Identifier: Apache-2.0
"""Engines for test_hybrid_gdn, one per process.

The reference is upstream SGLang's own Engine on the stock Qwen3.5 checkpoint.
The other serves the same weights through OmniScheduler under a test-only
architecture name, which upstream's per-architecture defaults do not know.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import sglang
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.runtime_context import get_schedule
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import AutoTokenizer

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.bootstrap import InfrastructureOptions
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    SGLangGenerationEngineBuilder,
)
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput
from tests.unit_test.scheduling.test_hybrid_gdn import (
    EngineSpec,
    FreePoolCounts,
    OmniRun,
    ReferenceRun,
    RequestRun,
    Scenario,
    ScenarioRun,
    TopTwoCandidates,
)

TEST_ARCH = "HybridGdnTestForCausalLM"
SCENARIO_TIMEOUT_SECONDS = 300.0
IDLE_TIMEOUT_SECONDS = 30.0
OUTBOX_POLL_SECONDS = 0.05

logger = logging.getLogger(__name__)


class HybridGdnTestModel(Qwen3_5ForConditionalGeneration):
    """Qwen3.5's text backbone under an architecture name upstream does not know."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # language_model_only builds no vision tower to load these into.
        return super().load_weights(
            (name, tensor) for name, tensor in weights if "visual" not in name
        )


@dataclass(frozen=True, kw_only=True)
class GenerationJob:
    input_ids: list[int]
    max_new_tokens: int


@dataclass(frozen=True, kw_only=True)
class SlotUsage:
    """One scheduler step's view of a request and of the shared state pool."""

    output_token_count: int
    mamba_slot_id: int
    occupied_mamba_slot_count: int


class GreedyRequestBuilder:
    """Turns a GenerationJob payload into a greedy SGLang request."""

    def __init__(self, *, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, payload: StagePayload) -> SGLangARRequestData:
        job = payload.data
        assert isinstance(job, GenerationJob)
        sampling_params = SamplingParams(
            max_new_tokens=job.max_new_tokens, temperature=0.0, ignore_eos=True
        )
        sampling_params.normalize(tokenizer=None)
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=job.input_ids,
            sampling_params=sampling_params,
            vocab_size=self.vocab_size,
        )
        return SGLangARRequestData(
            req=req,
            input_ids=torch.tensor(job.input_ids, dtype=torch.long),
            stage_payload=payload,
            max_new_tokens=job.max_new_tokens,
            temperature=0.0,
        )


class RequestResultAdapter(Protocol):
    def __call__(self, request_data: SGLangARRequestData, /) -> StagePayload: ...


def request_result(request_data: SGLangARRequestData) -> StagePayload:
    payload = request_data.stage_payload
    payload.data = list(request_data.output_ids)
    return payload


class SlotUsageReporter:
    """Stream builder reporting each step's slot and how many slots are taken."""

    def __init__(self) -> None:
        self.mamba_allocator: MambaSlotAllocator | None = None

    def __call__(
        self,
        request_id: str,
        request_data: SGLangARRequestData,
        request_output: RequestOutput,
    ) -> list[OutgoingMessage]:
        """Stream-builder contract; the step output itself is not needed here."""
        mamba_allocator = self.mamba_allocator
        assert mamba_allocator is not None
        req = request_data.req
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=SlotUsage(
                    output_token_count=len(req.output_ids),
                    mamba_slot_id=int(req.kv.mamba_pool_idx),
                    occupied_mamba_slot_count=mamba_allocator.size
                    - mamba_allocator.available_size(),
                ),
            )
        ]


class HybridGdnTestEngineBuilder(SGLangGenerationEngineBuilder[SGLangARRequestData]):
    """Plain greedy text generation; everything hybrid comes from the framework."""

    model_name = "hybrid-gdn-test"

    def __init__(self, *, spec: EngineSpec, shim_dir: Path) -> None:
        self.spec = spec
        self.shim_dir = shim_dir
        self.context_length = spec.engine.context_length
        self.slot_usage_reporter = SlotUsageReporter()

    def resolve_checkpoint(self, model_path: str) -> str:
        """Serve the stock weights text-only under TEST_ARCH."""
        source = Path(model_path)
        for entry in source.iterdir():
            if entry.name != "config.json":
                (self.shim_dir / entry.name).symlink_to(entry.resolve())
            else:
                pass
        config = json.loads((source / "config.json").read_text())
        config["architectures"] = [TEST_ARCH]
        config["language_model_only"] = True
        (self.shim_dir / "config.json").write_text(json.dumps(config))
        return str(self.shim_dir)

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "max_running_requests": self.spec.engine.max_running_requests,
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "trust_remote_code": False,
            "dtype": dtype,
        }

    def infra_kwargs(self) -> InfrastructureOptions:
        return {"total_gpu_memory_fraction": self.spec.total_gpu_memory_fraction}

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> ModelRunner[SGLangARRequestData]:
        return ModelRunner(model_worker, output_proc)

    def make_adapters(
        self, model: HybridGdnTestModel
    ) -> tuple[GreedyRequestBuilder, RequestResultAdapter]:
        return GreedyRequestBuilder(vocab_size=model.config.vocab_size), request_result

    def extra_scheduler_kwargs(self) -> SchedulerExtras[SGLangARRequestData]:
        return {"stream_output_builder": self.slot_usage_reporter}

    def post_scheduler_setup(
        self,
        scheduler: OmniScheduler[SGLangARRequestData],
        model_runner: ModelRunner[SGLangARRequestData],
    ) -> None:
        """Builder hook contract; only the scheduler's state pool is needed here."""
        pool = scheduler.req_to_token_pool
        assert isinstance(pool, HybridReqToTokenPool)
        self.slot_usage_reporter.mamba_allocator = pool.mamba_allocator


@dataclass(kw_only=True)
class RequestRecord:
    prompt_name: str
    max_new_tokens: int
    aborted: bool = False
    output_token_counts: list[int] = field(default_factory=list)
    occupied_mamba_slot_counts: list[int] = field(default_factory=list)
    mamba_slot_ids: list[int] = field(default_factory=list)
    result_output_ids: list[list[int]] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class ScenarioProgress:
    """What a scenario left behind; request snapshots wait for the final drain."""

    name: str
    request_ids: list[str]
    idle_free_counts: FreePoolCounts
    output_token_count_at_abort: int | None
    result_count_at_abort: int | None


class SchedulerDriver:
    """Feeds OmniScheduler's inbox and files its outbox per request."""

    def __init__(
        self,
        scheduler: OmniScheduler[SGLangARRequestData],
        prompt_token_ids: dict[str, list[int]],
    ) -> None:
        self.scheduler = scheduler
        self.prompt_token_ids = prompt_token_ids
        self.records: dict[str, RequestRecord] = {}
        self.failures: list[BaseException] = []
        # Daemon so a wedged scheduler cannot keep the engine process alive.
        self.thread = threading.Thread(
            target=self.serve, name="omni-scheduler", daemon=True
        )

    def serve(self) -> None:
        try:
            self.scheduler.start()
        except BaseException as error:
            self.failures.append(error)
            raise

    def start(self) -> None:
        self.thread.start()

    def stop_and_drain(self) -> None:
        """Stop the scheduler, then file every message it left in the outbox."""
        self.scheduler.stop()
        self.thread.join(timeout=SCENARIO_TIMEOUT_SECONDS)
        if self.thread.is_alive():
            raise RuntimeError("OmniScheduler thread did not stop")
        elif self.failures:
            raise RuntimeError("OmniScheduler thread failed") from self.failures[0]
        else:
            pass
        while True:
            try:
                message = self.scheduler.outbox.get_nowait()
            except queue.Empty:
                return
            self.record(message)

    def submit(self, request_id: str, prompt_name: str, max_new_tokens: int) -> str:
        self.records[request_id] = RequestRecord(
            prompt_name=prompt_name, max_new_tokens=max_new_tokens
        )
        payload = StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs=prompt_name),
            data=GenerationJob(
                input_ids=self.prompt_token_ids[prompt_name],
                max_new_tokens=max_new_tokens,
            ),
        )
        self.scheduler.inbox.put(
            IncomingMessage(request_id=request_id, type="new_request", data=payload)
        )
        return request_id

    def abort(self, request_id: str) -> None:
        self.records[request_id].aborted = True
        self.scheduler.inbox.put(IncomingMessage(request_id=request_id, type="abort"))

    def record(self, message: OutgoingMessage) -> None:
        record = self.records[message.request_id]
        if message.type == "stream":
            usage = message.data
            assert isinstance(usage, SlotUsage)
            record.output_token_counts.append(usage.output_token_count)
            record.occupied_mamba_slot_counts.append(usage.occupied_mamba_slot_count)
            record.mamba_slot_ids.append(usage.mamba_slot_id)
        elif message.type == "result":
            payload = message.data
            assert isinstance(payload, StagePayload)
            record.result_output_ids.append(list(payload.data))
        elif message.type == "error":
            record.error_messages.append(repr(message.data))
        else:
            raise RuntimeError(f"unexpected outbox message {message.type!r}")

    def pump(self) -> None:
        """File the next outbox message, if one arrives within a poll interval."""
        if self.failures:
            raise RuntimeError("OmniScheduler thread failed") from self.failures[0]
        elif not self.thread.is_alive():
            raise RuntimeError("OmniScheduler thread exited")
        else:
            pass
        try:
            message = self.scheduler.outbox.get(timeout=OUTBOX_POLL_SECONDS)
        except queue.Empty:
            return
        self.record(message)

    def wait_for_results(self, request_ids: list[str]) -> None:
        deadline = time.monotonic() + SCENARIO_TIMEOUT_SECONDS
        while any(
            not self.records[request_id].result_output_ids
            and not self.records[request_id].error_messages
            for request_id in request_ids
        ):
            if time.monotonic() > deadline:
                raise TimeoutError(f"requests did not finish: {request_ids}")
            else:
                pass
            self.pump()

    def wait_for_output_token_count(self, request_id: str, token_count: int) -> None:
        deadline = time.monotonic() + SCENARIO_TIMEOUT_SECONDS
        record = self.records[request_id]
        while (
            not record.output_token_counts
            or record.output_token_counts[-1] < token_count
        ):
            if record.result_output_ids or record.error_messages:
                raise RuntimeError(f"{request_id} ended before {token_count} tokens")
            elif time.monotonic() > deadline:
                raise TimeoutError(f"{request_id} never reached {token_count} tokens")
            else:
                pass
            self.pump()

    def free_pool_counts(self) -> FreePoolCounts:
        pool = self.scheduler.req_to_token_pool
        assert isinstance(pool, HybridReqToTokenPool)
        return FreePoolCounts(
            request_row_count=pool.available_size(),
            mamba_slot_count=pool.mamba_allocator.available_size(),
            kv_token_count=self.scheduler.token_to_kv_pool_allocator.available_size(),
        )

    def wait_for_idle(self, baseline: FreePoolCounts) -> FreePoolCounts:
        """Free counts once every slot is back, or the last counts on timeout."""
        deadline = time.monotonic() + IDLE_TIMEOUT_SECONDS
        free_counts = self.free_pool_counts()
        while free_counts != baseline and time.monotonic() < deadline:
            self.pump()
            free_counts = self.free_pool_counts()
        return free_counts

    def request_run(self, request_id: str) -> RequestRun:
        record = self.records[request_id]
        return RequestRun(
            request_id=request_id,
            prompt_name=record.prompt_name,
            max_new_tokens=record.max_new_tokens,
            aborted=record.aborted,
            output_token_counts=record.output_token_counts,
            occupied_mamba_slot_counts=record.occupied_mamba_slot_counts,
            mamba_slot_ids=record.mamba_slot_ids,
            result_output_ids=record.result_output_ids,
            error_messages=record.error_messages,
        )


def run_scenario(
    driver: SchedulerDriver,
    spec: EngineSpec,
    scenario: Scenario,
    baseline: FreePoolCounts,
) -> ScenarioProgress:
    request_ids: list[str] = []
    output_token_count_at_abort = None
    result_count_at_abort = None
    if scenario.abort_prompt_name is None:
        abort_id = None
    else:
        abort_id = driver.submit(
            f"{scenario.name}-target-{scenario.abort_prompt_name}",
            scenario.abort_prompt_name,
            spec.abort_max_new_tokens,
        )
        request_ids.append(abort_id)
    for wave_index, wave in enumerate(scenario.waves):
        wave_ids = [
            driver.submit(
                f"{scenario.name}-{wave_index}-{position}-{prompt_name}",
                prompt_name,
                spec.max_new_tokens,
            )
            for position, prompt_name in enumerate(wave)
        ]
        request_ids.extend(wave_ids)
        if abort_id is not None and wave_index == 0:
            driver.wait_for_output_token_count(
                abort_id, spec.abort_after_output_token_count
            )
            target = driver.records[abort_id]
            output_token_count_at_abort = target.output_token_counts[-1]
            result_count_at_abort = len(target.result_output_ids)
            driver.abort(abort_id)
        else:
            pass
        driver.wait_for_results(wave_ids)
    idle_free_counts = driver.wait_for_idle(baseline)
    logger.info(f"scenario {scenario.name}: free pools after drain {idle_free_counts}")
    return ScenarioProgress(
        name=scenario.name,
        request_ids=request_ids,
        idle_free_counts=idle_free_counts,
        output_token_count_at_abort=output_token_count_at_abort,
        result_count_at_abort=result_count_at_abort,
    )


def run_omni(spec: EngineSpec) -> OmniRun:
    ModelRegistry.models[TEST_ARCH] = HybridGdnTestModel
    with tempfile.TemporaryDirectory(prefix="hybrid-gdn-shim-") as shim_dir:
        builder = HybridGdnTestEngineBuilder(spec=spec, shim_dir=Path(shim_dir))
        scheduler = builder.build(
            spec.model_dir,
            dtype=spec.engine.dtype,
            server_args_overrides=spec.engine.model_dump(
                exclude={"context_length", "dtype"}
            ),
        )
        pool = scheduler.req_to_token_pool
        assert isinstance(pool, HybridReqToTokenPool)
        logger.info(
            f"hybrid GDN test engine: max_mamba_cache_size="
            f"{get_schedule().max_mamba_cache_size} "
            f"max_running_requests={scheduler.max_running_requests} "
            f"request_row_count={pool.size} "
            f"mamba_slot_count={pool.mamba_allocator.size} "
            f"max_total_num_tokens={scheduler.max_total_num_tokens} "
            f"chunked_prefill_size={scheduler.chunked_prefill_size}"
        )
        driver = SchedulerDriver(scheduler, spec.prompt_token_ids)
        baseline = driver.free_pool_counts()
        driver.start()
        try:
            progress = [
                run_scenario(driver, spec, scenario, baseline)
                for scenario in spec.scenarios
            ]
            pool_stats = scheduler.pool_stats_observer.get_pool_stats()
        finally:
            driver.stop_and_drain()
        return OmniRun(
            max_mamba_cache_size=get_schedule().max_mamba_cache_size,
            max_running_requests=scheduler.max_running_requests,
            request_row_count=pool.size,
            mamba_slot_count=pool.mamba_allocator.size,
            max_total_num_tokens=scheduler.max_total_num_tokens,
            chunked_prefill_size=scheduler.chunked_prefill_size,
            is_hybrid_ssm=pool_stats.is_hybrid_ssm,
            idle_used_mamba_slot_count=pool_stats.mamba_num_used,
            baseline_free_counts=baseline,
            scenarios=[
                ScenarioRun(
                    name=finished.name,
                    requests=[
                        driver.request_run(request_id)
                        for request_id in finished.request_ids
                    ],
                    idle_free_counts=finished.idle_free_counts,
                    output_token_count_at_abort=finished.output_token_count_at_abort,
                    result_count_at_abort=finished.result_count_at_abort,
                )
                for finished in progress
            ],
        )


def run_reference(spec: EngineSpec) -> ReferenceRun:
    tokenizer = AutoTokenizer.from_pretrained(spec.model_dir)
    prompt_token_ids = {
        prompt_name: tokenizer.encode(text, add_special_tokens=False)
        for prompt_name, text in spec.prompts.items()
    }
    prompt_names = list(prompt_token_ids)
    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": spec.max_new_tokens,
        "ignore_eos": True,
    }
    engine = sglang.Engine(
        model_path=spec.model_dir,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        log_level="info",
        **spec.engine.model_dump(),
    )
    try:
        single_outputs = {
            prompt_name: engine.generate(
                input_ids=prompt_token_ids[prompt_name],
                sampling_params=sampling_params,
                return_logprob=True,
                top_logprobs_num=2,
            )
            for prompt_name in prompt_names
        }
        batched_outputs = engine.generate(
            input_ids=[prompt_token_ids[prompt_name] for prompt_name in prompt_names],
            sampling_params=sampling_params,
        )
        repeat_output_ids = {
            prompt_name: engine.generate(
                input_ids=prompt_token_ids[prompt_name],
                sampling_params=sampling_params,
            )["output_ids"]
            for prompt_name in prompt_names
        }
    finally:
        engine.shutdown()
    return ReferenceRun(
        prompt_token_ids=prompt_token_ids,
        single_output_ids={
            prompt_name: output["output_ids"]
            for prompt_name, output in single_outputs.items()
        },
        # Upstream lists each position's top logprobs as (logprob, token id, text).
        single_top_two={
            prompt_name: [
                TopTwoCandidates(
                    token_ids=(candidates[0][1], candidates[1][1]),
                    logprobs=(candidates[0][0], candidates[1][0]),
                )
                for candidates in output["meta_info"]["output_top_logprobs"]
            ]
            for prompt_name, output in single_outputs.items()
        },
        repeat_output_ids=repeat_output_ids,
        batched_output_ids={
            prompt_name: output["output_ids"]
            for prompt_name, output in zip(prompt_names, batched_outputs)
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["reference", "omni"])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    logging.getLogger().setLevel(logging.INFO)
    spec = EngineSpec.model_validate_json(args.spec.read_text())
    if args.engine == "reference":
        run = run_reference(spec)
    else:
        run = run_omni(spec)
    args.output.write_text(run.model_dump_json())


if __name__ == "__main__":
    main()
