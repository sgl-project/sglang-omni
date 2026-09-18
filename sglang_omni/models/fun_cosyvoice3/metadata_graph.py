# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
from functools import partial

logger = logging.getLogger(__name__)

_MODULE = "sglang.srt.model_executor.runner.metadata_glue_graph"


class _ModuleView:
    def __init__(self, module, **overrides):
        self._module = module
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._module, name)


def patch_metadata_capture(prefill_backend: str, decode_backend: str) -> None:
    from sglang.srt.environ import envs

    flag = getattr(envs, "SGLANG_ENABLE_METADATA_GLUE_GRAPH", None)
    if flag is None:
        return
    # note (0xtoward): Only FA3 metadata replay has been validated.
    flag.set(flag.get() and prefill_backend == decode_backend == "fa3")
    if not flag.get():
        return
    try:
        module = importlib.import_module(_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != _MODULE:
            raise
        logger.warning("Metadata glue graph is unavailable in this SGLang version")
        return
    if isinstance(module.torch, _ModuleView):
        return

    # note (0xtoward): Other threads must not invalidate metadata capture.
    cuda = module.torch.cuda
    module.torch = _ModuleView(
        module.torch,
        cuda=_ModuleView(
            cuda, graph=partial(cuda.graph, capture_error_mode="thread_local")
        ),
    )
