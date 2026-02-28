from __future__ import annotations
from typing import List, Dict, Any
from sglang_omni.vendor.sglang.core import (
    Req
)


class PrefillAdder:
    def __init__(self, prefill_budget: int):
        self.prefill_budget = prefill_budget
        self.current_prefill_count = 0
    
    def add_one_requests(self, req):
        pass
    
    def try_allocate_one(self, req):
        pass

class PrefillManager:
    def __init__(
        self,
    ):
        self.waiting_queue = []
        
    def add_one_request(self, req):
        #TODO(ocs884): whether to use the Req cls from sglang
        self.waiting_queue.append(req)
    
    def schedule_next_batch(self, prefill_budget: int):
        # Implement the logic to schedule the next batch of tasks based on the waiting queue
        if len(self.waiting_queue) == 0:
            return None

        adder = PrefillAdder(prefill_budget)
        
        # Get requests from the waiting queue to a new prefill batch
        can_run_list = []
        for waiting_req in self.waiting_queue:
            if req := adder.try_allocate_one(waiting_req):
                can_run_list.append(req)
            else:
                #TODO(ocss884): maybe distinguish AddReqResult
                break
        # Batch dataclass of req for return
        return
        
    def prefill(self):
        # Implement the logic to prefill the scheduler with the specified number of samples
        pass