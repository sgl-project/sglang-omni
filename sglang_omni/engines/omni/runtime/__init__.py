# SPDX-License-Identifier: Apache-2.0
"""Runtime module - model-type-specific batching and I/O logic."""

from .ar import (
    ARBatchData,
    ARBatchPlanner,
    ARInputPreparer,
    AROutputProcessor,
    ARRequestData,
    ARResourceManager,
)
from .common import (
    EosIterationController,
    SimpleResourceManager,
    SinglePassIterationController,
)
from .encoder import (
    EncoderBatchData,
    EncoderBatchPlanner,
    EncoderInputPreparer,
    EncoderOutputProcessor,
    EncoderRequestData,
)
from .interfaces import (
    BatchPlanner,
    CacheManager,
    InputPreparer,
    IterationController,
    OutputProcessor,
    ResourceManager,
)

_SGLANG_EXPORTS = {
    "SGLangARRequestData",
    "SGLangBatchPlanner",
    "SGLangResourceManager",
    "SGLangOutputProcessor",
    "SGLangIterationController",
    "SGLangModelRunner",
}

_TALKER_EXPORTS = {
    "TalkerARRequestData",
    "TalkerBatchPlanner",
    "TalkerResourceManager",
    "TalkerOutputProcessor",
    "TalkerIterationController",
    "TalkerModelRunner",
}


class _SGLangAttr:
    """Lazy proxy for attributes provided by the sglang_ar module."""

    def __init__(self, name: str):
        self._name = name
        self._value = None

    def _resolve(self):
        if self._value is None:
            # Delegate to module-level __getattr__ for actual resolution
            self._value = globals()["__getattr__"](self._name)
        return self._value

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)


class _TalkerAttr:
    """Lazy proxy for attributes provided by the sglang_talker module."""

    def __init__(self, name: str):
        self._name = name
        self._value = None

    def _resolve(self):
        if self._value is None:
            # Delegate to module-level __getattr__ for actual resolution
            self._value = globals()["__getattr__"](self._name)
        return self._value

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)


# Provide module-level bindings for lazily loaded SGLang exports.
SGLangARRequestData = _SGLangAttr("SGLangARRequestData")
SGLangBatchPlanner = _SGLangAttr("SGLangBatchPlanner")
SGLangResourceManager = _SGLangAttr("SGLangResourceManager")
SGLangOutputProcessor = _SGLangAttr("SGLangOutputProcessor")
SGLangIterationController = _SGLangAttr("SGLangIterationController")
SGLangModelRunner = _SGLangAttr("SGLangModelRunner")

# Provide module-level bindings for lazily loaded Talker exports.
TalkerARRequestData = _TalkerAttr("TalkerARRequestData")
TalkerBatchPlanner = _TalkerAttr("TalkerBatchPlanner")
TalkerResourceManager = _TalkerAttr("TalkerResourceManager")
TalkerOutputProcessor = _TalkerAttr("TalkerOutputProcessor")
TalkerIterationController = _TalkerAttr("TalkerIterationController")
TalkerModelRunner = _TalkerAttr("TalkerModelRunner")


__all__ = [
    # Protocols
    "BatchPlanner",
    "CacheManager",
    "ResourceManager",
    "IterationController",
    "InputPreparer",
    "OutputProcessor",
    "SimpleResourceManager",
    "SinglePassIterationController",
    "EosIterationController",
    # Encoder
    "EncoderRequestData",
    "EncoderBatchData",
    "EncoderBatchPlanner",
    "EncoderInputPreparer",
    "EncoderOutputProcessor",
    # AR (Simple)
    "ARRequestData",
    "ARBatchData",
    "ARBatchPlanner",
    "ARResourceManager",
    "ARInputPreparer",
    "AROutputProcessor",
    # AR (SGLang)
    "SGLangARRequestData",
    "SGLangBatchPlanner",
    "SGLangResourceManager",
    "SGLangOutputProcessor",
    "SGLangIterationController",
    "SGLangModelRunner",
    # Talker
    "TalkerARRequestData",
    "TalkerBatchPlanner",
    "TalkerResourceManager",
    "TalkerOutputProcessor",
    "TalkerIterationController",
    "TalkerModelRunner",
]


def __getattr__(name: str):
    if name in _SGLANG_EXPORTS:
        from . import sglang_ar

        return getattr(sglang_ar, name)
    if name in _TALKER_EXPORTS:
        from . import sglang_talker

        return getattr(sglang_talker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
