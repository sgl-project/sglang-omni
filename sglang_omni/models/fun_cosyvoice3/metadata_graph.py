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


def patch_metadata_capture() -> None:
    from sglang.srt.environ import envs

    flag = getattr(envs, "SGLANG_ENABLE_METADATA_GLUE_GRAPH", None)
    if flag is None or not flag.get():
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

    # Global capture aborts when any thread submits unsafe CUDA work, and this
    # process also runs preprocessing. Scope the override to metadata prep.
    cuda = module.torch.cuda
    module.torch = _ModuleView(
        module.torch,
        cuda=_ModuleView(
            cuda, graph=partial(cuda.graph, capture_error_mode="thread_local")
        ),
    )
