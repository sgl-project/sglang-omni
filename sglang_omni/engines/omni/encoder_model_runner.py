# SPDX-License-Identifier: Apache-2.0
"""EncoderModelRunner - stateless model executor."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from .types import ModelRunnerOutput, RequestOutput, SchedulerOutput

if TYPE_CHECKING:
    from .runtime.interfaces import InputPreparer, OutputProcessor

logger = logging.getLogger(__name__)


class EncoderModelRunner:
    """ModelRunner for encoder models (pure inference, no cache logic)."""

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

        # Prepare inputs
        model_inputs = self.input_preparer.prepare(scheduler_output, self.device)

        # Execute model
        with torch.inference_mode():
            model_output = self.model(**model_inputs)

        # Process outputs
        outputs: dict[str, RequestOutput] = self.output_processor.process(
            model_output, scheduler_output
        )

        # Build metadata
        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )
