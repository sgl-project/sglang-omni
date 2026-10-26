# SPDX-License-Identifier: Apache-2.0
"""Warm the session LM through its prompt and one-row extend paths."""

import logging
import time
from dataclasses import replace

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import DynamicGradMode

from sglang_omni.models.personaplex.architecture import SILENCE_CODES
from sglang_omni.models.personaplex.model_runner import PersonaPlexModelRunner
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.request_builders import build_lm_request
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest

WARMUP_FRAMES = 2

logger = logging.getLogger(__name__)


class PersonaPlexSessionScheduler(OmniScheduler[SGLangARRequestData]):
    model_runner: PersonaPlexModelRunner

    @DynamicGradMode()
    def warm_up_serving_thread(self) -> None:
        """Run every supported batch size before the scheduler can report ready."""
        model_runner = self.model_runner
        device = model_runner.model_device
        max_batch_size = model_runner.model.fusion_buffer.shape[0]
        started_seconds = time.perf_counter()
        caller_codes = torch.tensor(SILENCE_CODES).repeat(WARMUP_FRAMES, 1)
        rng_devices = [device.index or 0] if device.type != "cpu" else []
        with torch.random.fork_rng(devices=rng_devices, device_type=device.type):
            for batch_size in range(1, max_batch_size + 1):
                requests: list[SchedulerRequest] = []
                try:
                    for index in range(batch_size):
                        request_id = f"personaplex-warmup-{batch_size}-{index}"
                        payload = StagePayload(
                            request_id,
                            request=OmniRequest(inputs=None),
                            data=PersonaPlexState(
                                user_codes=caller_codes,
                            ).to_dict(),
                        )
                        request_data = build_lm_request(
                            payload,
                            vocab_size=model_runner.model.config.vocab_size,
                            context_length=self.model_config.context_len,
                        )
                        OmniScheduler.normalize_req_token_arrays(request_data.req)
                        request_data.req.omni_data = request_data
                        request_data.req.skip_radix_cache_insert = True
                        requests.append(
                            SchedulerRequest(request_id=request_id, data=request_data)
                        )

                    # Note (edwardzh): Both paths sample [batch_size, vocabulary] logits, so each seed mode needs only one forward.
                    for step, sampling_seed in enumerate((None, 0)):
                        for request in requests:
                            inputs = request.data.talker_model_inputs
                            sampling = replace(inputs["sampling"], seed=sampling_seed)
                            inputs["sampling"] = sampling
                            request.data.req.sampling_params.sampling_seed = (
                                sampling.text_seed
                            )
                            request.data.req.init_next_round_input()
                            request.data.req.extend_end = request.data.req.seqlen
                        batch = ScheduleBatch.init_new(
                            reqs=[request.data.req for request in requests],
                            req_to_token_pool=self.req_to_token_pool,
                            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                            tree_cache=self.tree_cache,
                            model_config=self.model_config,
                            enable_overlap=False,
                            spec_algorithm=SpeculativeAlgorithm.NONE,
                        )
                        batch.prepare_for_extend()
                        output = model_runner.execute(
                            SchedulerOutput(
                                requests=requests, batch_data=batch, step_id=step
                            )
                        )
                        for request in requests:
                            request.data.req.output_ids.append(
                                int(output.outputs[request.request_id].data)
                            )
                            request.data.req.prefix_len = (
                                request.data.req.kv.kv_committed_len
                            )
                finally:
                    if device.type != "cpu":
                        torch.accelerator.synchronize(device)
                    else:
                        pass
                    for request in requests:
                        temporary_request = request.data.req
                        if not temporary_request.kv.holds_kv:
                            pass
                        elif temporary_request.kv.kv_allocated_len == 0:
                            self.req_to_token_pool.free(temporary_request)
                        else:
                            release_kv_cache(
                                temporary_request,
                                self.tree_cache,
                                checkpoint=False,
                            )
        elapsed_seconds = time.perf_counter() - started_seconds
        logger.info(
            f"PersonaPlex LM warmup completed for batch sizes 1..{max_batch_size} "
            f"with seeded and unseeded sampling in {elapsed_seconds:.2f}s"
        )
