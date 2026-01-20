# SPDX-License-Identifier: Apache-2.0
"""Generic ModelRunner - stateless model executor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .types import ModelRunnerOutput, SchedulerOutput

if TYPE_CHECKING:
    from .runtime.interfaces import InputPreparer, OutputProcessor


class ModelRunner:
    """Generic model executor.

    Responsibilities:
    - Convert SchedulerOutput to model inputs (via InputPreparer)
    - Execute model forward pass
    - Convert model outputs to RequestOutputs (via OutputProcessor)

    Completely stateless. All state lives in Scheduler.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_preparer: InputPreparer,
        output_processor: OutputProcessor,
        device: torch.device | str = "cuda",
    ):
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.input_preparer = input_preparer
        self.output_processor = output_processor

        # Move model to device and set to eval mode
        self.model = model.to(device)
        self.model.eval()

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute model on batch."""
        # 1. Prepare inputs (model-specific)
        model_inputs = self.input_preparer.prepare(scheduler_output, self.device)

        # 2. Forward pass
        with torch.inference_mode():
            model_output = self.model(**model_inputs)

        # 3. Process outputs (model-specific)
        request_outputs = self.output_processor.process(
            model_output,
            scheduler_output,
        )

        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}
        return ModelRunnerOutput(
            outputs=request_outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )
