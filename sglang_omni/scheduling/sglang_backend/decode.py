import logging
from typing import Callable

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()


class DecodeManager:
    def __init__(
        self,
        server_args: ServerArgs,
        token_to_kv_pool_allocator,
        on_retract: Callable[[Req], None],
    ):
        self.server_args = server_args
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.on_retract = on_retract
        self.new_token_ratio = 1.0
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)

    def schedule_next_batch(
        self,
        forward_ct: int,
    ):
        if not self.runnable:
            return None

        initial_bs = self.running_batch.batch_size()

        if (kv_full_retract_flag := not self.running_batch.check_decode_mem()) or (
            TEST_RETRACT and forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            old_available_tokens = self.token_to_kv_pool_allocator.available_size()
            old_ratio = self.new_token_ratio
            retracted_reqs, new_token_ratio, reqs_to_abort = (
                self.running_batch.retract_decode(self.server_args)
            )
            new_available_tokens = self.token_to_kv_pool_allocator.available_size()
            new_token_gained = new_available_tokens - old_available_tokens

            self.num_retracted_reqs = len(retracted_reqs)

            # TODO:(ocss884) enable metrics
            # if self.enable_metrics and len(retracted_reqs) > 0:
            #     self.metrics_collector.increment_retracted_reqs(
            #         num_retracted_reqs=len(retracted_reqs),
            #         num_retracted_input_tokens=sum(
            #             len(r.origin_input_ids) for r in retracted_reqs
            #         ),
            #         num_retracted_output_tokens=sum(
            #             len(r.output_ids) for r in retracted_reqs
            #         ),
            #     )
            self.new_token_ratio = new_token_ratio

            # TODO(ocss884): implement abort reqs
            # for req in reqs_to_abort:

            msg_prefix = (
                "KV cache pool is full. Retract requests. "
                if kv_full_retract_flag
                else "Testing retraction. "
            )
            msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
            if kv_full_retract_flag:
                msg_details += (
                    f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
                )
            logger.warning(msg_prefix + msg_details)

            for req in retracted_reqs:
                self.on_retract(req)

        if self.running_batch.batch_size() < initial_bs:
            self.running_batch.batch_is_full = False
        self.running_batch.prepare_for_decode()
        return self.running_batch

    @property
    def runnable(self) -> bool:
        return self.running_batch.batch_size() > 0
