# SPDX-License-Identifier: Apache-2.0
"""Load the packaged SenseNova model and tokenizer."""
from __future__ import annotations

import torch
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


def load_model_and_tokenizer(
    model_path: str,
    server_args: ServerArgs,
) -> dict[str, torch.nn.Module | PreTrainedTokenizerBase]:
    model_dtype = PRECISION_TO_TYPE.get(
        server_args.pipeline_config.model_precision, torch.bfloat16
    )
    model_load_options: dict[str, str | bool | torch.dtype] = {
        "torch_dtype": model_dtype
    }
    if server_args.trust_remote_code:
        model_load_options["trust_remote_code"] = True
    else:
        pass
    if server_args.revision is not None:
        model_load_options["revision"] = server_args.revision
    else:
        pass

    tokenizer_load_options: dict[str, str | bool] = {}
    if server_args.trust_remote_code:
        tokenizer_load_options["trust_remote_code"] = True
    else:
        pass
    if server_args.revision is not None:
        tokenizer_load_options["revision"] = server_args.revision
    else:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_load_options)
    model = AutoModel.from_pretrained(model_path, **model_load_options).eval()
    model_device = get_local_torch_device()
    current_platform.set_device(model_device)
    model = model.to(model_device)
    return {"model": model, "tokenizer": tokenizer}
