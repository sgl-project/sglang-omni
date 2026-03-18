import logging

from sglang_omni.vendor.sglang.core import ModelConfig, ScheduleBatch, ServerArgs, envs

from ..model_worker import ModelWorker, ModelWorkerConfig
from .cache import CacheManager
from .decode import DecodeManager
from .prefill import PrefillManager

logger = logging.getLogger(__name__)


class SchedulerConfig:
    server_args = ServerArgs
    model_config = ModelConfig
    device = int


def _conver_model_worker_config(config: SchedulerConfig) -> ModelWorkerConfig:
    return ModelWorkerConfig()


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
    ):
        self.config = config
        self.server_args = self.config.server_args
        self.device = self.config.device

        # The current forward batch
        self.cur_batch = None
        # The last forward batch
        self.last_batch = None

        # Init memory pool and cache manager
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )

        self.cache_manager = CacheManager()

        self.init_chunked_prefill()

        # Init prefill & decode manager
        self.prefill_manager = PrefillManager(
            page_size=self.server_args.page_size,
            chunked_prefill_size=self.chunked_prefill_size,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.cache_manager.tree_cache,
            model_config=self.tp_worker.model_config,
            enable_overlap=False,
        )
        self.decode_manager = DecodeManager(
            server_args=self.server_args,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        )

        self.forward_ct = 0
        self.init_schedule_policy()

    def init_schedule_policy(self):
        # Init schedule policy and new token estimation
        # Enable preemption for priority scheduling.
        self.init_new_token_ratio = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * self.server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        self.new_token_ratio = self.init_new_token_ratio

    def _init_model_worker(self):
        config = _conver_model_worker_config(self.config)
        self.tp_worker = ModelWorker(
            config=config,
            server_args=self.server_args,
            gpu_id=self.device,
            tp_rank=0,
        )

    def init_chunked_prefill(self):
        # TODO(ocss884): For simplicity, we disabled `dynamic chunking` and `mixed chunk` for now
        self.chunked_prefill_size = self.server_args.chunked_prefill_size

    def init_overlap(self):
        # TODO(ocss884): implement overlap scheduling
        raise NotImplementedError

    def recv_requests():
        pass

    def normal_loop(self) -> None:
        """A normal scheduler loop."""
        while True:
            recv_reqs = self.recv_requests()
            self.process_requess(recv_reqs)

            batch = self.get_next_batch_to_run(batch)
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)

            self.last_batch = batch

    def process_batch_result(self, batch, result):
        # TODO(ocss884): implement result processing, including sending output to tokenizer and updating cache
        pass

    def get_next_batch_to_run(self):
        # TODO(ocss884): maybe support more scheduling strategies

        chunked_req_to_exclude = set()

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            # For prefill-only batch, we can avoid going through decoding step.
            if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)

        running_bs = len(self.decode_manager.running_batch)
        num_allocatable_reqs = self.get_num_allocatable_reqs(running_bs)

        if (
            next_batch := self.prefill_manager.schedule_next_batch(
                self.decode_manager.running_batch, num_allocatable_reqs
            )
            is not None
        ):
            ret = next_batch
        elif self.decode_manager.runnable:
            ret = self.decode_manager.schedule_next_batch(self.forward_ct)
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )
        else:
            ret = None

        return ret

    def get_num_allocatable_reqs(self, running_bs):
        # NOTE(ocss884): cp from sglang but removed pp
        res = self.server_args.max_running_requests - running_bs
        return res

    # TODO(ocss884): fix it
    def run_batch(batch: ScheduleBatch):
        raise NotImplementedError


def launch_scheduler(config: SchedulerConfig):
    scheduler = Scheduler(config)
    return scheduler
