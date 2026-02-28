import logging
import torch
from torch.cuda import Stream as CudaStream
from torch.cuda import StreamContext as CudaStreamContext
from .prefill import PrefillManager
from .decode import DecodeManager
from sglang_omni.vendor.sglang.core import (
    ServerArgs,
    ModelConfig
)

logger = logging.getLogger(__name__)

class SchedulerConfig:
    pass

class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        server_args, ServerArgs,
        device
    ):
        self.device  = device
        self.prefill_manager = PrefillManager()
        self.decode_manager = DecodeManager()
    
    def init_overlap(self):
        self.device_module = torch.get_device_module(self.device)
        self.default_stream: CudaStream = self.
        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
    def receive_msg():
    
    def normal_loop(self) -> None:
        """A normal scheduler loop."""
        while True:
            recv_reqs = self.recv_requests()
            self.process_requess(recv_reqs)
            
            batch = self.get_next_batch_to_run()
            
        
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        
        scheduler_batch = self
        
    def _schedule_next_batch(self):
        # Implement the logic to schedule the next batch of tasks based on the state of the prefill and decode managers
        pass
    
    def event_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)


def launch_scheduler(config: SchedulerConfig):
    scheduler = Scheduler(config)
    scheduler.event_loop()
    return scheduler