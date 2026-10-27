# SPDX-License-Identifier: Apache-2.0
"""Install lightweight sglang stubs so CSM unit tests run without the wheel."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace


class FakeReq:
    def __init__(
        self,
        *,
        rid,
        origin_input_text,
        origin_input_ids,
        sampling_params,
        vocab_size=None,
        extra_key=None,
        **kwargs,
    ) -> None:
        del kwargs
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids = origin_input_ids
        self.sampling_params = sampling_params
        self.vocab_size = vocab_size
        self.extra_key = extra_key
        self._codec_suppress_tokens = None
        self._input_embeds_are_projected = False
        self.extend_range = SimpleNamespace(length=len(origin_input_ids))


class FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.stop_strs = []
        self.stop_regex_strs = []

    def normalize(self, tokenizer) -> None:
        del tokenizer
        if self.stop_strs is None:
            self.stop_strs = []
        if self.stop_regex_strs is None:
            self.stop_regex_strs = []


def ensure_sglang_stubs() -> None:
    """No-op when the real sglang package is importable."""
    try:
        import sglang  # noqa: F401

        return
    except ImportError:
        pass

    if "sglang.srt.managers.schedule_batch" in sys.modules:
        return

    def _mod(name: str, *, package: bool = True) -> types.ModuleType:
        module = types.ModuleType(name)
        if package:
            module.__path__ = []
            module.__package__ = name
        sys.modules[name] = module
        return module

    sglang = _mod("sglang")
    srt = _mod("sglang.srt")
    sglang.srt = srt
    managers = _mod("sglang.srt.managers")
    srt.managers = managers
    schedule_batch = _mod("sglang.srt.managers.schedule_batch")
    managers.schedule_batch = schedule_batch
    sampling = _mod("sglang.srt.sampling")
    srt.sampling = sampling
    sampling_params = _mod("sglang.srt.sampling.sampling_params")
    sampling.sampling_params = sampling_params
    utils = _mod("sglang.srt.utils")
    srt.utils = utils
    models = _mod("sglang.srt.models")
    srt.models = models
    llama = _mod("sglang.srt.models.llama")
    models.llama = llama
    logits = _mod("sglang.srt.layers")
    srt.layers = logits
    logits_processor = _mod("sglang.srt.layers.logits_processor")
    logits.logits_processor = logits_processor

    schedule_batch.Req = FakeReq
    schedule_batch.FINISH_MATCHED_TOKEN = lambda matched: ("matched", matched)
    sampling_params.SamplingParams = FakeSamplingParams
    utils.add_prefix = lambda name, prefix: f"{prefix}.{name}" if prefix else name

    class _Llama:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    llama.LlamaForCausalLM = _Llama
    logits_processor.LogitsProcessorOutput = object
