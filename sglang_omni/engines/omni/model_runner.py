# SPDX-License-Identifier: Apache-2.0
"""ModelRunner - stateless model executor."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from .types import ModelRunnerOutput, RequestOutput, SchedulerOutput

if TYPE_CHECKING:
    from .runtime.interfaces import InputPreparer, OutputProcessor

logger = logging.getLogger(__name__)


class ModelRunner:
    """Generic stateless model executor.

    Responsibilities:
    - Convert SchedulerOutput to model inputs (via InputPreparer)
    - Execute model forward pass
    - Convert model outputs to RequestOutputs (via OutputProcessor)

    Works with any model type (encoder, AR, DiT, etc.) - the model-specific
    logic is handled by InputPreparer and OutputProcessor.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_preparer: InputPreparer,
        output_processor: OutputProcessor,
        *,
        device: torch.device | str = "cuda",
    ):
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.input_preparer = input_preparer
        self.output_processor = output_processor

        self.model = model.to(device)
        self.model.eval()

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute model inference for given scheduler output."""
        if scheduler_output.num_requests == 0:
            return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})

        # 1. Prepare inputs (model-specific via InputPreparer)
        model_inputs = self.input_preparer.prepare(scheduler_output, self.device)

        # 2. Forward pass
        with torch.inference_mode():
            model_output = self.model(**model_inputs)

        # 3. Process outputs (model-specific via OutputProcessor)
        outputs: dict[str, RequestOutput] = self.output_processor.process(
            model_output, scheduler_output
        )

        # 4. Build metadata
        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )
