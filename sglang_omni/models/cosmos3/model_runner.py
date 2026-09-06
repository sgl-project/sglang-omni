# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 thinker runner with precomputed vision embedding injection."""

from __future__ import annotations

from typing import Any

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner


class Cosmos3ThinkerModelRunner(ThinkerModelRunner):
    """Use the Qwen3-VL deepstack injection path with Cosmos3's model layout."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        # ThinkerModelRunner's execution logic is architecture-agnostic, but its
        # constructor unwraps Qwen3-Omni's ``model.thinker`` hierarchy. Cosmos3
        # exposes the language model directly, so bind the same fields here.
        ModelRunner.__init__(self, tp_worker, output_processor)
        self._outer_model = self.model
        self._text_model = self.model.model
        self._embed_tokens = self._text_model.embed_tokens
        self._th_host_bufs = None
        self._th_slot = 0
        self._should_capture_hidden = None

        config = tp_worker.model_runner.model_config.hf_config
        self._image_token_id = int(config.image_token_id)
        self._video_token_id = int(config.video_token_id)
        self._audio_token_id = -1

    def lookahead_eligible(self, batch: Any) -> bool:
        return ModelRunner.lookahead_eligible(self, batch)


__all__ = ["Cosmos3ThinkerModelRunner"]
