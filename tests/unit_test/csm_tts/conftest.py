# SPDX-License-Identifier: Apache-2.0
"""CPU test harness for csm_tts (CPU-only sglang stub).

When ``sglang`` is not installed (the CPU dev box), install a MINIMAL stub of
the four upstream symbols the csm_tts engine-side modules import at module
level, so ``request_builders`` / ``model_runner`` stay importable and their
pure-logic seams (clamp math, fingerprinting, finish marking, embed overlay)
are testable without a GPU. With real sglang present (container CI) the stub
is never installed and the real classes are exercised.

Stubbed contracts (kept 1:1 with the upstream call shapes used by csm_tts):

- ``sglang.srt.managers.schedule_batch.Req`` — kwargs ctor, ``rid`` /
  ``origin_input_ids`` / ``sampling_params`` / ``extra_key``; ``finished()``
  derives from ``finished_reason``; ``is_chunked = 0``.
- ``sglang.srt.managers.schedule_batch.FINISH_MATCHED_TOKEN`` — ``matched``.
- ``sglang.srt.sampling.sampling_params.SamplingParams`` — kwargs incl.
  ``sampling_seed``; ``normalize(tokenizer)`` sets ``stop_strs``.
- ``sglang.srt.layers.logits_processor.LogitsProcessorOutput``.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any


def _install_sglang_stub() -> None:
    try:
        import sglang.srt.managers.schedule_batch  # noqa: F401

        return  # real sglang available — never stub
    except ImportError:
        pass

    class FINISH_MATCHED_TOKEN:  # noqa: N801 — upstream name
        def __init__(self, matched: Any) -> None:
            self.matched = matched

    class Req:
        def __init__(
            self,
            rid: str = "",
            origin_input_text: str = "",
            origin_input_ids: list[int] | None = None,
            sampling_params: Any = None,
            vocab_size: int | None = None,
            extra_key: str | None = None,
            **kwargs: Any,
        ) -> None:
            self.rid = rid
            self.origin_input_text = origin_input_text
            self.origin_input_ids = list(origin_input_ids or [])
            self.sampling_params = sampling_params
            self.vocab_size = vocab_size
            self.extra_key = extra_key
            self.output_ids: list[int] = []
            self.finished_reason: Any = None
            self.is_chunked = 0

        def finished(self) -> bool:
            return self.finished_reason is not None

    class SamplingParams:
        def __init__(
            self,
            max_new_tokens: int = 128,
            temperature: float = 1.0,
            top_p: float = 1.0,
            top_k: int = -1,
            sampling_seed: int | None = None,
            **kwargs: Any,
        ) -> None:
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
            self.sampling_seed = sampling_seed
            self.stop_strs: Any = None
            self.stop_regex_strs: Any = None

        def normalize(self, tokenizer: Any) -> None:
            del tokenizer
            self.stop_strs = []
            self.stop_regex_strs = []

    class LogitsProcessorOutput:
        def __init__(self, next_token_logits: Any = None, **kwargs: Any) -> None:
            self.next_token_logits = next_token_logits
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _module(name: str, **attrs: Any) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__ = []  # mark as package for submodule imports
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    class _Placeholder:
        """Inert stand-in for upstream classes that csm_tts only references
        through the sglang_backend package import chain (never constructed in
        these CPU tests)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    _module("sglang")
    _module("sglang.srt")

    class _Env:
        """Any ``envs.NAME.get()`` returns a falsy default."""

        def __getattr__(self, name: str) -> Any:
            return SimpleNamespace(get=lambda: False)

    _module("sglang.srt.environ", envs=_Env())
    _module("sglang.srt.managers")
    _module(
        "sglang.srt.managers.schedule_batch",
        Req=Req,
        ScheduleBatch=_Placeholder,
        FINISH_MATCHED_TOKEN=FINISH_MATCHED_TOKEN,
    )
    _module("sglang.srt.managers.schedule_policy", PrefillAdder=_Placeholder)
    _module("sglang.srt.mem_cache")
    _module("sglang.srt.mem_cache.cache_init_params", CacheInitParams=_Placeholder)
    _module("sglang.srt.mem_cache.radix_cache", RadixCache=_Placeholder)
    _module("sglang.srt.server_args", ServerArgs=_Placeholder)
    _module("sglang.srt.speculative")
    _module(
        "sglang.srt.speculative.spec_info",
        SpeculativeAlgorithm=SimpleNamespace(NONE=None),
    )
    _module("sglang.srt.models")
    _module("sglang.srt.models.llama", LlamaForCausalLM=_Placeholder)
    _module(
        "sglang.srt.utils",
        add_prefix=lambda name, prefix: f"{prefix}.{name}" if prefix else name,
    )
    _module("sglang.srt.sampling")
    _module("sglang.srt.sampling.sampling_params", SamplingParams=SamplingParams)
    _module("sglang.srt.layers")
    _module(
        "sglang.srt.layers.logits_processor",
        LogitsProcessorOutput=LogitsProcessorOutput,
    )


_install_sglang_stub()
