from __future__ import annotations

import atexit
import json
import shutil
import tempfile
from pathlib import Path

from sglang.kernels.ops.mamba.triton_ops import (
    initialize_mamba_selective_state_update_backend,
)
from transformers import AutoTokenizer

from sglang_omni.models.nemotron_voicechat.hf_config import (
    VOICECHAT_MODEL_ARCH_OVERRIDE,
    NemotronVoiceChatConfig,
    register_voicechat_hf_config,
)
from sglang_omni.models.nemotron_voicechat.model_runner import (
    NemotronVoiceChatModelRunner,
)
from sglang_omni.models.nemotron_voicechat.request_builders import (
    SYSTEM_PROMPT,
    apply_talker_result,
    apply_thinker_result,
    build_talker_request,
    build_thinker_request,
    talker_stream_output_builder,
    thinker_stream_output_builder,
)
from sglang_omni.models.nemotron_voicechat.talker import TALKER_ARCH
from sglang_omni.models.nemotron_voicechat.talker_model_runner import (
    NemotronVoiceChatTalkerModelRunner,
)
from sglang_omni.models.nemotron_voicechat.talker_scheduler import (
    NemotronTalkerScheduler,
)
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

TALKER_SPEAKER = "Aria"
TALKER_PROMPT_FRAMES = 37
TALKER_PLACEHOLDER_VOCAB = 8


def _shim_dir(name: str, source: Path) -> Path:
    source = source.resolve()
    shim = Path(tempfile.mkdtemp(prefix=f"sglang-omni-{name}-"))
    atexit.register(shutil.rmtree, shim, ignore_errors=True)
    for entry in source.iterdir():
        if entry.name != "config.json":
            (shim / entry.name).symlink_to(entry)
    return shim


def _talker_config(source: Path) -> dict:
    speech = json.loads((source / "config.json").read_text())["model"][
        "speech_generation"
    ]["model"]
    backbone = speech["tts_config"]["backbone_config"]
    return {
        "model_type": "gemma3_text",
        "architectures": [TALKER_ARCH],
        "hidden_size": backbone["hidden_size"],
        "intermediate_size": backbone["intermediate_size"],
        "num_hidden_layers": backbone["num_hidden_layers"],
        "num_attention_heads": backbone["num_attention_heads"],
        "num_key_value_heads": backbone["num_key_value_heads"],
        "head_dim": backbone["head_dim"],
        "sliding_window": backbone["sliding_window"],
        "attention_dropout": 0.0,
        "vocab_size": TALKER_PLACEHOLDER_VOCAB,
        "torch_dtype": "bfloat16",
        "nemotron_speech": {
            "tts_config": speech["tts_config"],
            "codec_config": speech["codec_config"],
            "inference_top_p_or_k": speech["inference_top_p_or_k"],
            "inference_noise_scale": speech["inference_noise_scale"],
            "inference_force_speech_silence_on_eos": speech[
                "inference_force_speech_silence_on_eos"
            ],
            "tokenizer_name": speech["tts_config"]["cas_config"][
                "pretrained_tokenizer_name"
            ],
            "bos_token": speech.get("bos_token"),
            "eos_token": speech.get("eos_token"),
            "pad_token": speech.get("pad_token"),
            "text_vocab_size": 131072,
            "char_vocab_size": 256,
            "speaker": TALKER_SPEAKER,
            "prompt_frames": TALKER_PROMPT_FRAMES,
        },
    }


class _VoiceChatEngineBuilder(TtsEngineBuilder):
    scheduler_class: type

    def __init__(self, *, max_running_requests: int = 1) -> None:
        self.max_running_requests = max_running_requests

    def generation_defaults(self, *, dtype):
        defaults = {
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_running_requests": self.max_running_requests,
            "chunked_prefill_size": -1,
            "dtype": dtype,
            "trust_remote_code": False,
        }
        return defaults

    def setup_model(self, *, model_worker, checkpoint_dir, device, gpu_id, server_args):
        del model_worker, checkpoint_dir, device, gpu_id, server_args

    def make_scheduler(self, **kwargs):
        extra = (
            kwargs.pop("extra_scheduler_kwargs", None) or self.extra_scheduler_kwargs()
        )
        return self.scheduler_class(
            tp_worker=kwargs.pop("model_worker"),
            abort_callback=self.make_abort_callback(),
            request_finished_callback=self.make_request_finished_callback(),
            **kwargs,
            **extra,
        )


class NemotronVoiceChatEngineBuilder(_VoiceChatEngineBuilder):
    model_name = "nemotron-voicechat"
    context_length = 8192
    scheduler_class = OmniScheduler

    def __init__(self, *, max_running_requests: int = 1) -> None:
        super().__init__(max_running_requests=max_running_requests)
        self.model_arch_override = VOICECHAT_MODEL_ARCH_OVERRIDE
        self._source: Path | None = None

    def resolve_checkpoint(self, model_path):
        source = Path(resolve_model_path(model_path))
        self._source = source
        shim = _shim_dir("voicechat", source)
        config = NemotronVoiceChatConfig.from_dict(
            json.loads((source / "config.json").read_text())
        )
        (shim / "config.json").write_text(config.to_json_string())
        return str(shim)

    def _prompt_tokens(self) -> tuple[list[int], int]:
        """The instruction a conversation opens with, and the padding id.

        Which token means begin, end and pad is the checkpoint's to say — the
        three are easy to confuse here, since this tokenizer's padding token is
        the frame-locked silence marker rather than anything called "pad".
        """
        stt = json.loads((self._source / "config.json").read_text())["model"]["stt"][
            "model"
        ]
        tokenizer_name = stt.get("pretrained_llm", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except OSError as exc:
            raise OSError(
                f"Failed to load Thinker tokenizer from {tokenizer_name!r}. "
                "Check the path or Hub access. "
                "For offline use, download the tokenizer files first."
            ) from exc
        ident = tokenizer.convert_tokens_to_ids
        prompt = [
            ident(stt.get("bos_token", "<s>")),
            *tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False),
            ident(stt.get("eos_token", "</s>")),
        ]
        return prompt, ident(stt.get("pad_token", "<SPECIAL_12>"))

    def pre_infra_setup(self, checkpoint_dir):
        del checkpoint_dir
        register_voicechat_hf_config()

    def customize_server_args(self, server_args):
        initialize_mamba_selective_state_update_backend(server_args)

    def make_model_runner(self, model_worker, output_proc):
        return NemotronVoiceChatModelRunner(model_worker, output_proc)

    def make_adapters(self, model):
        vocab_size = int(model.llm.config.vocab_size)

        prompt_token_ids, pad_token_id = self._prompt_tokens()
        self._model_runner_pad_id = pad_token_id

        def build(payload):
            return build_thinker_request(
                payload,
                vocab_size=vocab_size,
                prompt_token_ids=prompt_token_ids,
                pad_token_id=pad_token_id,
            )

        return build, apply_thinker_result

    def extra_scheduler_kwargs(self):
        return {"stream_output_builder": thinker_stream_output_builder}


class NemotronVoiceChatTalkerEngineBuilder(_VoiceChatEngineBuilder):
    model_name = "nemotron-voicechat-talker"
    context_length = 4096
    scheduler_class = NemotronTalkerScheduler

    def __init__(
        self, *, max_running_requests: int = 1, context_length: int | None = None
    ) -> None:
        super().__init__(max_running_requests=max_running_requests)
        self.model_arch_override = TALKER_ARCH
        if context_length is not None:
            self.context_length = int(context_length)

    def resolve_checkpoint(self, model_path):
        source = Path(resolve_model_path(model_path))
        shim = _shim_dir("voicechat-talker", source)
        (shim / "config.json").write_text(json.dumps(_talker_config(source)))
        return str(shim)

    def generation_defaults(self, *, dtype):
        defaults = super().generation_defaults(dtype=dtype or "bfloat16")
        defaults["mem_fraction_static"] = 0.35
        return defaults

    def make_model_runner(self, model_worker, output_proc):
        return NemotronVoiceChatTalkerModelRunner(model_worker, output_proc)

    def make_adapters(self, model):
        vocab_size = int(model.config.vocab_size)
        prompt_frames = int(model.audio_prompt_latent.shape[0])

        def build(payload):
            return build_talker_request(
                payload, vocab_size=vocab_size, prompt_frames=prompt_frames
            )

        return build, apply_talker_result

    def extra_scheduler_kwargs(self):
        return {"stream_output_builder": talker_stream_output_builder}
